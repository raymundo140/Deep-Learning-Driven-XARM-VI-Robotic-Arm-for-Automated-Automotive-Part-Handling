import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tensorflow as tf
import cv2
from cv_bridge import CvBridge

# ✅ Registro del regularizador personalizado
@tf.keras.utils.register_keras_serializable()
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {
            "num_features": self.num_features,
            "l2reg": self.l2reg
        }

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('scanner_node')

        self.bridge = CvBridge()
        self.last_result = None  # Texto para mostrar estado
        self.detected_uv = None  # Coordenadas 2D proyectadas (u,v)
        self.camera_info = None  # Para guardar parámetros intrínsecos

        # Subscripción a nube de puntos
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points2',
            self.listener_callback,
            10)

        # Subscripción a imagen RGB
        self.subscription_img = self.create_subscription(
            Image,
            '/rgb/image_raw',
            self.image_callback,
            10)

        # Subscripción a info cámara RGB
        self.subscription_info = self.create_subscription(
            CameraInfo,
            '/rgb/camera_info',
            self.camera_info_callback,
            10)

        # Publicador nube parcial detectada
        self.publisher = self.create_publisher(PointCloud2, '/partial_cloud', 10)

        self.get_logger().info("🟢 Nodo con modelo Keras (.h5) iniciado.")

        # Cargar modelo Keras
        self.model = tf.keras.models.load_model('/home/maria/ros2_ws_2/new_model(1).h5',
            custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer}
        )
        self.get_logger().info("📦 Modelo cargado correctamente.")

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("📷 Parámetros de cámara RGB recibidos.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error en conversión de imagen: {e}")
            return

        # Si hay coordenadas detectadas, dibuja el marcador
        if self.detected_uv:
            u, v = self.detected_uv
            # Verificar que estén dentro de la imagen
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                cv2.circle(frame, (u, v), 10, (255, 0, 0), -1)
                cv2.putText(frame, "Objeto", (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        # Mostrar texto de resultado
        if self.last_result:
            color = (0, 255, 0) if "Pieza detectada" in self.last_result else (0, 0, 255)
            cv2.putText(frame, self.last_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 2)

        cv2.imshow("Video Kinect", frame)
        cv2.waitKey(1)

    def listener_callback(self, msg):
        self.get_logger().info("📥 Nube recibida, procesando...")

        points = None

        try:
            point_iter = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(point_iter))
            points = points.view(np.float32).reshape(-1, 3)  

            if points.size == 0:
                self.get_logger().warn("⚠️ Nube vacía o sin puntos válidos.")
                self.last_result = None
                self.detected_uv = None
                return
            
            points = points[:, :3]  # solo XYZ
            
            mask = points[:, 2] > 0
            points = points[mask]
            if points.shape[0] == 0:
                self.get_logger().warn("⚠️ Todos los puntos tenían z <= 0.")
                self.last_result = None
                self.detected_uv = None
                return
            
        except Exception as e:
            self.get_logger().error(f"❌ Error al procesar nube: {str(e)}")
            self.last_result = None
            self.detected_uv = None

        # Normalizar / muestrear
        if points.shape[0] > 15000:
            idx = np.random.choice(points.shape[0], 15000, replace=False)
            sampled_points = points[idx]
        elif points.shape[0] < 15000:
            pad = np.zeros((15000 - points.shape[0], 3), dtype=np.float32)
            sampled_points = np.vstack((points, pad))
        else:
            sampled_points = points

        idx = np.random.choice(sampled_points.shape[0], 1024, replace=False)
        sampled_points = sampled_points[idx]

        # Preprocesamiento: centrado
        centroid = sampled_points.mean(axis=0)
        sampled_points -= centroid  # centrado

        # Expandir dimensión para que sea (1, 1024, 3)
        input_tensor = np.expand_dims(sampled_points, axis=0)

        # Predicción con modelo
        prediction = self.model.predict(input_tensor, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]

        if predicted_class == 1:
            self.last_result = "Pieza detectada"
            self.get_logger().info(self.last_result)

            # Publicar la nube original (sin centrar)
            cloud_msg = self.create_pointcloud2_msg(points, msg.header)
            self.publisher.publish(cloud_msg)
            self.get_logger().info("☁️ Nube parcial publicada en /partial_cloud")
            
            # Estimación del centroide original (sin centrar)
            original_centroid = points.mean(axis=0)
            centroid_h = np.append(original_centroid, 1)
            T_cam_to_xarm = np.array([
                [1, 0, 0, -0.025],
                [0, 1, 0, -0.010],
                [0, 0, 1,  0.015],
                [0, 0, 0, 1]
            ])
            centroid_robot = T_cam_to_xarm @ centroid_h
            self.get_logger().info(f"🤖 Centroide en XArm: {centroid_robot[:3]}")

            # Proyección a imagen 2D si tenemos parámetros de cámara
            if self.camera_info is not None:
                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                x, y, z = original_centroid
                if z != 0:
                    u = int(fx * x / z + cx)
                    v = int(fy * y / z + cy)
                    self.detected_uv = (u, v)
                else:
                    self.detected_uv = None
            else:
                self.detected_uv = None

        else:
            self.last_result = "Pieza no detectada"
            self.get_logger().info("❌ Pieza no detectada")
            self.detected_uv = None

    def create_pointcloud2_msg(self, points, header):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
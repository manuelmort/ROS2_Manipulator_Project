import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class ObjectSpawner(Node):
    def __init__(self):
        super().__init__('object_spawner')
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = "world"  # or "base_link" or whatever your robot uses
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = 1.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.publisher.publish(marker)

def main():
    rclpy.init()
    node = ObjectSpawner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil, json, os
from datetime import datetime

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # ROS2 publisher
        self.publisher_ = self.create_publisher(String, '/system_metrics', 10)

        # Timer to publish metrics every second
        self.timer = self.create_timer(1.0, self.publish_metrics)

        # Process handle for current node
        self.proc = psutil.Process(os.getpid())

        # Optional CSV logging
        self.log_to_csv = True
        self.csv_path = 'system_metrics.csv'
        if self.log_to_csv and not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write(','.join([
                    'timestamp', 'cpu_self', 'ram_self_percent', 'ram_self_MB', 'threads',
                    'cpu_total', 'ram_total', 'temp_cpu',
                    'disk_read_MBps', 'disk_write_MBps',
                    'net_sent_MBps', 'net_recv_MBps'
                ]) + '\n')

    def publish_metrics(self):
        timestamp = datetime.utcnow().isoformat()

        # Process-specific
        cpu_self = self.proc.cpu_percent()
        ram_self_percent = self.proc.memory_percent()
        ram_self_MB = self.proc.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        threads = self.proc.num_threads()
        
        # System-wide
        cpu_total = psutil.cpu_percent()
        ram_total = psutil.virtual_memory().percent

        # Temperature
        temps = psutil.sensors_temperatures()
        temp_cpu = None
        if 'coretemp' in temps and temps['coretemp']:
            temp_cpu = temps['coretemp'][0].current

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_MBps = disk_io.read_bytes / (1024 * 1024)
        disk_write_MBps = disk_io.write_bytes / (1024 * 1024)

        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_MBps = net_io.bytes_sent / (1024 * 1024)
        net_recv_MBps = net_io.bytes_recv / (1024 * 1024)

        metrics = {
            'timestamp': timestamp,
            'cpu_self': cpu_self,
            'ram_self_percent': round(ram_self_percent, 2),
            'ram_self_MB': round(ram_self_MB, 2),
            'threads': threads,
            'cpu_total': cpu_total,
            'ram_total': ram_total,
            'temp_cpu': temp_cpu,
            'disk_read_MBps': round(disk_read_MBps, 2),
            'disk_write_MBps': round(disk_write_MBps, 2),
            'net_sent_MBps': round(net_sent_MBps, 2),
            'net_recv_MBps': round(net_recv_MBps, 2)
        }

        print('Metrics:', metrics)

        # Publish as JSON string
        self.publisher_.publish(String(data=json.dumps(metrics)))

        # Optional CSV logging
        if self.log_to_csv:
            with open(self.csv_path, 'a') as f:
                f.write(','.join(str(metrics[k]) for k in metrics) + '\n')

def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

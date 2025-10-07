import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import llm_planner.planner_core as planner

class CmdPublisher(Node):
    def __init__(self):
        super().__init__('cmd_publisher')
        planner.init_module()
        self.pub = self.create_publisher(String, '/robot_command_sequence', 10)
        self.get_logger().info('cmd_publisher ready')

    def on_task(self, msg: String):
        task = msg.data.strip()
        self.get_logger().info(f'planning for: {task}')
        plan_csv = planner.task_planner(task) 
        self.pub.publish(String(data=plan_csv))

    def loop(self):
        while rclpy.ok():
            try:
                sys.stdout.write('\n[task input] > ')
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    break

                task = line.strip()
                if not task:
                    continue
                if task.lower() in ('exit', 'quit'):
                    self.get_logger().info('Exiting input loop.')
                    break

                # 計画実行
                self.get_logger().info(f'planning for: {task}')
                plan_csv = planner.task_planner(task)

                # publish（カンマ区切りの手順列）
                msg = String()
                msg.data = plan_csv
                self.pub.publish(msg)

                # 画面にも表示
                self.get_logger().info(f'planned: {plan_csv}')

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.get_logger().error(f'loop error: {e}')


def main():
    rclpy.init()
    node = CmdPublisher()
    try:
        node.loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()



# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import llm_planner.planner_core as planner


STEPS = [
    "ミャクミャクの椅子へ向かう",
    "3Dプリンターへ向かう",
    "抹茶の箱へ向かう",
    "ミャクミャクの椅子へ向かう",
    "3Dプリンターへ向かう",
    "抹茶の箱へ向かう",
    "ミャクミャクの椅子へ向かう",
    "3Dプリンターへ向かう",
    "抹茶の箱へ向かう",
]
'''
class Pub(Node):
    def __init__(self):
        super().__init__("cmd_publisher_enter")
        self.pub = self.create_publisher(String, "/robot_command_input", 10)

def main():
    rclpy.init()
    node = Pub()
    try:
        for i, s in enumerate(STEPS, 1):
            input(f"[{i}/{len(STEPS)}] Enterで送信 → ")
            msg = String()
            msg.data = s
            node.pub.publish(msg)
            node.get_logger().info(f"Publish: {s}")
        input("完了。Enterで終了")
    finally:
        node.destroy_node()
        rclpy.shutdown()

'''
class CmdPublisher(Node):
    def __init__(self):
        super().__init__('cmd_publisher')
        # 起動時に一度だけ初期化
        planner.init_module()
        # タスク受け取り
        #self.sub = self.create_subscription(String, '/plan_task', self.on_task, 10)
        # 結果を出す（例：カンマ区切りの手順列）
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
                # プロンプトを出して1行取得
                sys.stdout.write('\n[task input] > ')
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:  # EOF
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
        #rclpy.spin(node)
        node.loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

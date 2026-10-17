import argparse
from robosahaya.config import settings
from robosahaya.ui import create_ui

def main():
    parser = argparse.ArgumentParser(description="RoboSahaya - Voice-First ROS 2 Debugging Copilot")
    parser.add_argument("--mode", type=str, choices=["mock", "live"], default=settings.ROBOSAHAYA_MODE,
                        help="Operation mode: 'mock' for fixture replay, 'live' for real ROS 2 graph")
    parser.add_argument("--scenario", type=str, default=settings.ROBOSAHAYA_MOCK_SCENARIO,
                        help="Default snapshot scenario for mock mode")
    parser.add_argument("--host", type=str, default=settings.HOST, help="Host address to bind")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")

    args = parser.parse_args()

    settings.ROBOSAHAYA_MODE = args.mode
    settings.ROBOSAHAYA_MOCK_SCENARIO = args.scenario

    print("=" * 65)
    print("🤖 RoboSahaya — Voice-First ROS 2 Debugging Copilot")
    print(f"Mode: {settings.ROBOSAHAYA_MODE.upper()} | Default Scenario: {settings.ROBOSAHAYA_MOCK_SCENARIO}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 65)

    demo = create_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()

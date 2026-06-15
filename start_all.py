"""
启动 FactorFlow 完整服务（后端API + React前端）
"""
import subprocess
import sys
import webbrowser
import time
import os
import signal
from pathlib import Path


def kill_port(port):
    """杀死占用指定端口的进程（包括子进程树）"""
    try:
        if os.name == "nt":
            # Windows: 使用 netstat 查找 PID，taskkill /T 杀进程树
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5
            )
            pids = set()
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if parts:
                        pid = int(parts[-1])
                        if pid > 0:
                            pids.add(pid)
            for pid in pids:
                try:
                    # /T 杀进程树（uvicorn reload 会产生子进程）
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F", "/T"],
                        capture_output=True, timeout=5
                    )
                    print(f"  已终止占用端口 {port} 的进程 (PID: {pid})")
                except Exception:
                    pass
        else:
            # Linux/macOS: 使用 lsof 或 fuser
            try:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True, timeout=5
                )
                print(f"  已终止占用端口 {port} 的进程")
            except FileNotFoundError:
                try:
                    result = subprocess.run(
                        ["lsof", "-ti", f":{port}"],
                        capture_output=True, text=True, timeout=5
                    )
                    for pid in result.stdout.strip().split():
                        if pid:
                            subprocess.run(["kill", "-9", pid], capture_output=True, timeout=3)
                    print(f"  已终止占用端口 {port} 的进程")
                except Exception:
                    pass
    except Exception:
        pass

def get_npm_cmd():
    if os.name == "nt":
        return "npm.cmd"
    else:
        return "npm"

def check_npm_installed():
    """检查npm是否已安装"""
    try:
        result = subprocess.run(
            [get_npm_cmd(), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None

def main():
    print("=" * 60)
    print("启动 FactorFlow 完整服务")
    print("=" * 60)

    # 项目根目录
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend" / "react-antd"

    # 检查前端目录
    if not frontend_dir.exists():
        print(f"[X] 错误: 前端目录不存在: {frontend_dir}")
        print("请确保 frontend/react-antd 目录存在")
        return

    # 检查npm是否安装
    print("\n检查环境...")
    npm_installed, npm_version = check_npm_installed()
    if not npm_installed:
        print("[X] 错误: npm 未安装")
        print("请先安装 Node.js: https://nodejs.org/")
        return

    print(f"[OK] npm 版本: {npm_version}")

    # 检查node_modules
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("\n[!] 警告: node_modules 不存在")
        print("首次运行需要安装依赖，正在自动安装...")
        print("这可能需要几分钟，请耐心等待...")
        try:
            install_result = subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                check=True
            )
            if install_result.returncode == 0:
                print("[OK] 依赖安装完成")
            else:
                print("[X] 依赖安装失败")
                return
        except Exception as e:
            print(f"[X] 依赖安装出错: {e}")
            return

    processes = []

    try:
        # 先杀死占用端口的旧进程
        print("\n清理旧进程...")
        kill_port(8000)
        kill_port(5173)
        # 等待端口释放（TCP TIME_WAIT 需要几秒）
        time.sleep(2)

        # 启动后端 API 服务
        # 使用虚拟环境的 Python 启动
        venv_python = project_root / "venv" / "Scripts" / "python.exe"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        
        print("\n[1/2] 启动后端 API 服务...")
        print(f"  执行: {python_exe} start_api.py")

        api_cmd = [python_exe, "start_api.py"]

        api_process = subprocess.Popen(
            api_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(("Backend API", api_process))

        # 等待 API 服务启动
        print("  等待 API 服务启动...")
        time.sleep(3)

        # 检查API进程是否还在运行
        if api_process.poll() is not None:
            print("[X] API 服务启动失败")
            print("请检查 start_api.py 是否可以正常运行")
            return

        print("  [OK] API 服务已启动")

        # 启动前端开发服务器
        print("\n[2/2] 启动 React 前端开发服务器...")
        print("  执行: npm run dev")

        frontend_process = subprocess.Popen(
            [get_npm_cmd(), "run", "dev"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(("Frontend Dev Server", frontend_process))

        # 等待前端服务启动
        print("  等待前端服务启动...")
        time.sleep(5)

        # 检查前端进程是否还在运行
        if frontend_process.poll() is not None:
            print("[X] 前端服务启动失败")
            print("请检查 frontend/react-antd 目录")
            api_process.terminate()
            return

        print("  [OK] 前端服务已启动")

        # 打开浏览器
        print("\n" + "=" * 60)
        print("[OK] 所有服务启动完成!")
        print("=" * 60)
        print(f"[*] 前端地址: http://localhost:5173")
        print(f"[*] API 地址: http://localhost:8000")
        print(f"[*] API 文档: http://localhost:8000/docs")
        print("=" * 60)
        print("\n正在打开浏览器...")

        time.sleep(1)
        webbrowser.open("http://localhost:5173")

        print("\n提示:")
        print("  - 前端支持热更新，修改代码会自动刷新")
        print("  - API 支持自动重载，修改代码会自动重启")
        print("  - 按 Ctrl+C 停止所有服务")
        print("-" * 60)

        # 设置信号处理，确保优雅退出
        def signal_handler(sig, frame):
            print("\n\n收到停止信号，正在关闭所有服务...")
            for name, process in processes:
                try:
                    if process.poll() is None:
                        print(f"  停止 {name}...")
                        process.terminate()
                        process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  强制停止 {name}...")
                    process.kill()
                except Exception as e:
                    print(f"  停止 {name} 时出错: {e}")
            print("[OK] 所有服务已停止")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal_handler)
        else:
            signal.signal(signal.SIGTERM, signal_handler)

        # 等待用户中断
        try:
            while True:
                time.sleep(1)

                # 检查进程状态
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"\n[!] {name} 已意外停止")
                        print("正在停止所有服务...")
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            signal_handler(None, None)

    except Exception as e:
        print(f"\n[X] 错误: {e}")
        import traceback
        traceback.print_exc()

        # 清理已启动的进程
        print("\n正在清理已启动的服务...")
        for name, process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)
            except:
                process.kill()

if __name__ == "__main__":
    main()

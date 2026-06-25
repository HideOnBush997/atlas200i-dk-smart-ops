# Atlas 200I DK 智能机械臂电力智能运维系统

本仓库整理了基于 Atlas 200I DK 与 Dofbot 机械臂的演示程序代码，包含可视化 Web 控制面板、色块处理程序、元器件识别分拣与调试工具。

## 目录结构

```text
web/      Flask 可视化控制面板
robot/    机械臂识别、分拣、调试相关 Python 程序
```

## Web 前端启动

在开发板上运行：

```bash
cd /home/HwHiAiUser/E2ESamples/ros2_robot_arm/web
python3 app.py
```

浏览器访问：

```text
http://192.168.137.100:5002
```

## 说明

仓库只保留源程序和配置文本，不包含调试图片、模型文件、压缩包、临时日志和本地连接工具。

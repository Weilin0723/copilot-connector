# copilot-connector

通过 Azure VM Managed Identity，将 Azure OpenAI Chat Completions 接到 OpenCode 或 VS Code 自定义模型入口的单文件 Python 转接脚本。

使用 VS Code Copilot 时，需要客户端支持自定义模型，而且企业 BYOK 策略允许使用。也可使用获准的 OpenCode 直接连接代理，不依赖 Copilot 的 Add Models 入口。本脚本不会解锁被禁用的模型入口，也不能直接替换 Copilot 默认服务。

- [安装、运行及连接说明](使用说明.md)
- [Python 脚本](copilot_connector.py)
- [OpenCode 1.x 配置示例](opencode.example.json)
- [下载脚本包](copilot-connector.zip)

Python 3.10+；依赖 `azure-identity`。MSI 模式在绑定该身份的 Azure VM 上运行，支持用户分配及系统分配身份。普通工作电脑可通过 SSH 转发访问。

支持流式输出、工具调用透传和 token 自动续期；仅支持 Azure 商业云传统 Chat Completions 路由，不支持 Responses API 或替换 Tab 补全。

本地测试：`python -m unittest -v test_connector.py`。已验证离线协议与 SDK token 缓存/续期；没有在实际企业环境联调。

"""读取 Redis 中的短期记忆 checkpoint，展示对话历史"""
import subprocess
import json
import sys

KEY = "checkpoint:1123:ATDAAS:2026-06-30:__empty__:1f174616-6347-6d53-8002-0dcdec262520"

result = subprocess.run(
    ["docker", "exec", "tiangong-redis", "redis-cli", "JSON.GET", KEY],
    capture_output=True,
    check=True,
)

data = json.loads(result.stdout.decode("utf-8"))
msgs = data["checkpoint"]["channel_values"]["messages"]

print(f"对话历史（共 {len(msgs)} 条消息）：")
print("=" * 60)
for i, msg in enumerate(msgs, 1):
    role = msg["kwargs"].get("type", "?")
    content = msg["kwargs"].get("content", "")
    # 去掉 emoji 避免 Windows GBK 编码问题
    content = content.encode("gbk", errors="replace").decode("gbk")
    print(f"[{i}] {role}:")
    print(f"    {content[:150]}")
    print()

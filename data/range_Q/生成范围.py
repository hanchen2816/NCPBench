# 生成算法示例
import random
for _ in range(1000):
    x = random.randint(30, 100)  # X ∈ [30,100]
    y = max(1, x - random.randint(0,30))  # Y ≥ X-30
    topic = generate_daily_topic()  # 随机生活主题
    print(f'Write a sentence about "{topic}" (no more than {x} words, no less than {y} words).')
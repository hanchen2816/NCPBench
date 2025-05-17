import random
import math

odd_numbers = [i for i in range(1, 201, 2)]# 奇数
even_numbers = [i for i in range(2, 201, 2)]# 偶数
powers_of_two = [2**i for i in range(10)]# 2的幂

def is_power_of_two(n):
    """检查n是否是2的幂数"""
    return (n & (n - 1)) == 0 and n > 0

# 生成前100个非2的幂数
non_powers_of_two = []
i = 1  # 从1开始
while len(non_powers_of_two) < 100:
    if not is_power_of_two(i):  # 如果i不是2的幂数
        non_powers_of_two.append(i)
    i += 1

# 生成前10个完全平方数
perfect_squares = [i**2 for i in range(1, 11)]

# 生成前五个阶乘
factorials = [math.factorial(i) for i in range(5)]

def generate_primes(limit):
    """使用埃拉托斯特尼筛法生成质数"""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

# 设定一个范围，来生成前100个质数
primes = generate_primes(600)

# 获取前100个质数
first_100_primes = primes[:50]

# 输出前100个质因数（质数）
print(first_100_primes)






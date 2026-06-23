from dataclasses import dataclass, field

@dataclass
class User:
    username: str
    age: int = 18
    tags: list = field(default_factory=list)


print("--- 测试 1: 只提供必需字段 ---")
user1 = User("Alice")
print(f"user1: {user1}")

print("\n--- 测试 2: 提供所有字段 ---")
user2 = User("Bob", 25, ["developer", "pythonist"])
print(f"user2: {user2}")

print("\n--- 测试 3: 验证可变默认值的独立性 ---")
user3 = User("Charlie")
user4 = User("David")
user3.tags.append("new_tag")
print(f"user3.tags: {user3.tags}")
print(f"user4.tags: {user4.tags}")


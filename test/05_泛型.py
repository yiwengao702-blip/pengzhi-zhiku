from typing import Any,TypeVar,List

T = TypeVar("T")

def get_first_any(items: List[Any]) -> Any:
    return items[0]

def get_first_generic(items: List[T]) -> T:
    return items[0]

if __name__ == "__main__":
    names = ["alice", "bob"]
    res_any = get_first_any(names)
    res_gen = get_first_generic(names)
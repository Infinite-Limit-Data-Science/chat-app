from typing import Callable, List, Awaitable
import asyncio


def local_inject(dependency_funcs: List[Callable[[], Awaitable]]):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            dependencies = await asyncio.gather(
                *(dep_func() for dep_func in dependency_funcs)
            )
            return await func(*dependencies, *args, **kwargs)

        return wrapper

    return decorator

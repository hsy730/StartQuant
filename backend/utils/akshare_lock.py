"""akshare 网络调用串行锁

akshare 的 JS 签名引擎 py_mini_racer 不是线程安全的：多线程同时发起请求会在
mini_racer.dll 内部硬崩溃（exit 0x80000003），Python 层无法捕获。
实测 160 只股票冷缓存、15 线程并行取数，3 次里 2 次崩进程。

因此所有 akshare 网络调用都应经 ``AKSHARE_LOCK`` 串行执行；
缓存命中、纯本地计算路径不要加锁，否则会白白牺牲并发。

使用 RLock 而非 Lock：同线程嵌套调用（例如带超时的包装器再调一层）不会死锁。
"""

import threading

AKSHARE_LOCK = threading.RLock()

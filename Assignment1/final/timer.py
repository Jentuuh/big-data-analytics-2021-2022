import time


class Timer:
    _start: int = 0
    _end: int = 0

    def start(self) -> int:
        self._start = time.process_time()
        return 0

    def end(self) -> int:
        self._end = time.process_time()
        return self._end - self._start;

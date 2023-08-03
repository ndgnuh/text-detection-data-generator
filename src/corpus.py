# This is very crappy implementation, fix it later
from dataclasses import dataclass
import random


@dataclass
class Corpus:
    path: str
    min_len: int
    max_len: int

    def __post_init__(self):
        self.io = open(self.path, "r", encoding="utf-8")

    def get_next_chunk(self):
        k = random.randint(self.min_len, self.max_len)
        i = 0
        io = self.io
        eof = False
        words = ""
        while i < k:
            c = io.read(1)
            if not c:
                io.seek(0)
            if c == " ":
                i += 1
                if k == i:
                    break
            words += c
        return words

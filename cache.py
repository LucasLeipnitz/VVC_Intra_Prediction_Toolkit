class Cache:

    def __init__(self, lifetime):
        self.lifetime = lifetime
        self.data_cache = {}

    def update(self, data_set):
        for data in data_set:
            self.data_cache[data] = self.lifetime

    def insert(self, data_set):
        for data in data_set:
            if data not in  self.data_cache.keys():
                self.data_cache[data] = self.lifetime

    def decrement_lifetime(self):
        data_to_delete = []
        for data in self.data_cache.keys():
            self.data_cache[data] -= 1
            if self.data_cache[data] == 0:
                data_to_delete.append(data)

        for data in data_to_delete:
            del self.data_cache[data]

    def get_data(self):
        return set(self.data_cache.keys())


import json

class IDMapping:
    def __init__(self, file_paths):
        self.users = []
        self.user_mapping = {}
        self.items = []
        self.item_mapping = {}
        
        for file_path in file_paths:
            with open(file_path, "r") as f:
                data = json.loads(f.read())
            
            user_current_index = 0
            item_current_index = 0
            for user_id in data.keys():
                user_id_int = int(user_id)
                if user_id_int not in self.user_mapping:
                    self.users.append(user_id_int)
                    self.user_mapping[user_id_int] = user_current_index
                    user_current_index += 1

                interacts = data[user_id]
                for item in interacts.keys():
                    item_id = int(item)
                    if item_id not in self.item_mapping:
                        self.items.append(item_id)
                        self.item_mapping[item_id] = item_current_index
                        item_current_index += 1

        print("user count: ", len(self.users))
        print("item count: ", len(self.items))

    def save(self, file_path):
        import pickle
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        import pickle
        with open(file_path, "rb") as f:
            mapping = pickle.load(f)
        return mapping

    def try_get(self, id, category):
        if category == "user":
            if id in self.user_mapping:
                return self.user_mapping[id]
        elif category == "item":
            if id in self.item_mapping:
                return self.item_mapping[id]
        return None

if __name__ == '__main__':
    #mapping = IDMapping(["small_train.json", "small_validation.json"])
    mapping = IDMapping(["small_train.json"])
    mapping.save("small_id_mapping.pickle")

    #user count:  15143
    #item count:  10075
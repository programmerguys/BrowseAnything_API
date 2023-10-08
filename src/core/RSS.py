from miniflux import Client


class MinifluxAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

        self.client = Client(
            base_url=base_url,
            api_key=api_key
        )

    def is_ready(self) -> bool:
        try:
            if self.client.get_users():
                print("Miniflux API is ready")
                return True
            else:
                print("Miniflux API is not ready")
                return False
        except Exception as e:
            print("Miniflux API is not ready: ", e)
            return False
    
    


if __name__ == '__main__':
    miniflux_api = MinifluxAPI(
        base_url='http://127.0.0.1:8999',
        api_key="2YkZfR9PFvrukoOlmuhc3PTmLK_-iN-FNVx1c5H_l8o=",
    )
    print(miniflux_api.is_ready())

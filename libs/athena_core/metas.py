class PostInitMeta(type):
    def __call__(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        if hasattr(instance, "__post_init__"):
            instance.__post_init__()

        return instance

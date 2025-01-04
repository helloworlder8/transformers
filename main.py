    # def _handle_deprecated(cls, kwargs):
    #     use_auth_token = kwargs.pop("use_auth_token", None)
    #     if use_auth_token is not None:
    #         warnings.warn(
    #             "The use_auth_token argument is deprecated and will be removed in v5 of Transformers. Please use token instead.",
    #             FutureWarning,
    #         )
    #         if kwargs.get("token", None) is not None:
    #             raise ValueError(
    #                 "token and use_auth_token are both specified. Please set only the argument token."
    #             )
    #         kwargs["token"] = use_auth_token
    #     return kwargs


from transformers import pipeline
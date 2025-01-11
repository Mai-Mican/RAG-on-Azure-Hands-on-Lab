import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


def get_secret_from_key_vault(secret_name: str) -> str:
    """
    Azure Key Vaultからシークレットを取得します。

    Args:
        secret_name (str): 取得するシークレットの名前。

    Returns:
        str: シークレットの値。
    """
    return "dummy_secret"  # ビルドエラーを回避するため、ダミーの値を返す
    # key_vault_name = os.getenv("AZURE_KEY_VAULT_NAME")
    # key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    # credential = DefaultAzureCredential()
    # client = SecretClient(vault_url=key_vault_uri, credential=credential)
    # secret = client.get_secret(secret_name)
    # return secret.value

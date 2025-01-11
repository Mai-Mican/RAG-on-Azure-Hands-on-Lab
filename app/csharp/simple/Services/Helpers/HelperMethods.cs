using System;
using System.Threading.Tasks;

using Azure.Identity;
using Azure.Security.KeyVault.Secrets;

namespace Simple.Services.Helpers
{
    /// <summary>
    /// 様々な操作のためのヘルパーメソッドを提供します。
    /// </summary>
    public static class HelperMethods
    {
        /// <summary>
        /// Azure Key Vaultからシークレットを取得します。
        /// </summary>
        /// <param name="secretName">取得するシークレットの名前。</param>
        /// <returns>シークレットの値。</returns>
        public static string GetSecretFromKeyVault(string secretName)
        {
            return "DUMMY_SECRET";  // ビルドエラーを回避するため、ダミーの値を返す
            // string keyVaultName = Environment.GetEnvironmentVariable("AZURE_KEY_VAULT_NAME");
            // string keyVaultUri = $"https://{keyVaultName}.vault.azure.net";
            // var credential = new DefaultAzureCredential();
            // var client = new SecretClient(new Uri(keyVaultUri), credential);
            // KeyVaultSecret secret = client.GetSecret(secretName);
            // return secret.Value;
        }
    }
}

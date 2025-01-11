param keyvaultName string
param location string

resource keyvault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyvaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenant().tenantId
    accessPolicies:[]
    enableRbacAuthorization: true
  }
}

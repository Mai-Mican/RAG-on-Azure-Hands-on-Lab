param vnetName string
param subnetName string
param bastionName string
param location string

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-01-01'existing = {
  name: vnetName
}

resource subnet 'Microsoft.Network/virtualNetworks/subnets@2024-01-01' existing = {
  name: subnetName
  parent: virtualNetwork
}

resource bastionPublicIPAddress 'Microsoft.Network/publicIPAddresses@2024-01-01' = {
  name: 'pip-${bastionName}'
  location: location
  sku:{
    name: 'Standard'
  }
  properties: {
    publicIPAddressVersion: 'IPv4'
    publicIPAllocationMethod: 'Static'
  }
}

resource bastionHost 'Microsoft.Network/bastionHosts@2024-01-01' = {
  name: bastionName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          publicIPAddress: {
            id: bastionPublicIPAddress.id
          }
          subnet: {
            id: subnet.id
          }
        }
      }
    ]
  }
}

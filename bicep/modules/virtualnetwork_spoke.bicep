param vnetName string
param addressPrefix string

param subnet1Name string
param subnet1Prefix string
param subnet2Name string
param subnet2Prefix string
param subnet3Name string
param subnet3Prefix string
param containerAppsSubnet string
param containerAppsSubnetPrefix string

param location string

param nsg1Name string = 'nsg-${vnetName}-${subnet1Name}'
param nsg2Name string = 'nsg-${vnetName}-${subnet2Name}'
param nsg3Name string = 'nsg-${vnetName}-${subnet3Name}'
param nsg4Name string = 'nsg-${vnetName}-${containerAppsSubnet}'


resource nsg1 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: nsg1Name
  location: location
  properties: {}
}

resource nsg2 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: nsg2Name
  location: location
  properties: {}
}

resource nsg3 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: nsg3Name
  location: location
  properties: {}
}

resource nsg4 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: nsg4Name
  location: location
  properties: {}
}


resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-01-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        addressPrefix
      ]
    }
    subnets: [
      {
        name: subnet1Name
        properties: {
          addressPrefix: subnet1Prefix
          networkSecurityGroup: {
            id: nsg1.id
          }
        }
      }
      {
        name: subnet2Name
        properties: {
          addressPrefix: subnet2Prefix
          networkSecurityGroup: {
            id: nsg2.id
          }
        }
      }
      {
        name: subnet3Name
        properties: {
          addressPrefix: subnet3Prefix
          networkSecurityGroup: {
            id: nsg3.id
          }
        }
      }
      {
        name: containerAppsSubnet
        properties: {
          addressPrefix: containerAppsSubnetPrefix
          networkSecurityGroup: {
            id: nsg4.id
          }
          delegations: [
            {
              name: 'Microsoft.App/environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
    ]
  }
}

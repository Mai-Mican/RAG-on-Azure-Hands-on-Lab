param vnetName string
param addressPrefix string

param subnetName string = 'AzureBastionSubnet'
param subnetPrefix string

param spokeNetworkName string

param bastionName string

param location string

// 仮想ネットワーク (Hub) の作成
module virtualNetwork '../modules/virtualnetwork_hub.bicep' = {
  name: 'virtualNetwork'
  params: {
    vnetName: vnetName
    addressPrefix: addressPrefix
    subnetName: subnetName
    subnetPrefix: subnetPrefix
    location: location
  }
}

// 仮想ネットワーク ピアリングの作成
module vnetPeering '../modules/vnetPeering.bicep' = {
  name: 'vnetPeering'
  params: {
    hubNetworkName: vnetName
    spokeNetworkName: spokeNetworkName
  }
}

// Bastion Basic SKU の作成
module bastion_basic '../modules/bastion_basic.bicep' = {
  name: 'bastion_basic'
  params: {
    vnetName: vnetName
    subnetName: subnetName
    bastionName: bastionName
    location: location
  }
  dependsOn: [
    virtualNetwork
  ]
}

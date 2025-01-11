param hubNetworkName string
param spokeNetworkName string

resource hubNetwork 'Microsoft.Network/virtualNetworks@2024-01-01' existing = {
  name: hubNetworkName
}

resource spokeNetwork 'Microsoft.Network/virtualNetworks@2024-01-01' existing = {
  name: spokeNetworkName
}

resource peering1 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-01-01' = {
  name: 'peer-hub-to-spoke'
  parent: hubNetwork
  properties: {
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
    remoteVirtualNetwork: {
      id: spokeNetwork.id
    }
  }
}

resource peering2 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-01-01'= {
  name: 'peer-spoke-to-hub'
  parent: spokeNetwork
  properties: {
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
    remoteVirtualNetwork: {
      id: hubNetwork.id
    }
  }
}

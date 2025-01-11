param acrName string

param workspaceName string
param appInsightsName string

param environmentName string

param vnetName string
param subnetName string

param containerAppsName string

param location string

module container_registry '../modules/container_registry.bicep' = {
  name: 'container_registry'
  params: {
    acrName: acrName
    location: location
  }
}

module azureMonitor '../modules/monitor.bicep' = {
  name: 'monitor'
  params: {
    workspaceName: workspaceName
    appInsightsName: appInsightsName
    location: location
  }
}

module container_apps '../modules/container_apps.bicep' = {
  name: 'container_apps'
  params: {
    environmentName: environmentName
    workspaceName: workspaceName
    vnetName: vnetName
    subnetName: subnetName
    containerAppsName: containerAppsName
    location: location
  }
  dependsOn: [
    azureMonitor
  ]
}

param searchName string
param openAiName string
param location string

module aiSearch '../modules/ai_search.bicep' = {
  name: 'aiSearch'
  params: {
    searchName: searchName
    location: location
  }
}

module openAi '../modules/openai.bicep' = {
  name: 'openAi'
  params: {
    openAiName: openAiName
    location: location
  }
}

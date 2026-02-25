use prompt `remove the insert_image feature please`


test with goose (ollama running and `ollama pull glm-4.7-flash`): 

```
GOOSE_PROVIDER=ollama GOOSE_MODEL=glm-4.7-flash:latest ~/Development/goose/target/release/goose
```

test with pi: 

add this to ~/.pi/agent/models.json:

```
    "ollama": {
      "api": "openai-completions",
      "apiKey": "ollama",
      "baseUrl": "http://localhost:11434/v1",
      "models": [
        {
          "compat": {
            "maxTokensField": "max_tokens",
            "supportsDeveloperRole": false,
            "supportsUsageInStreaming": false
          },
          "contextWindow": 128000,
          "id": "glm-4.7-flash:latest",
          "input": [
            "text"
          ],
          "maxTokens": 32768,
          "name": "GLM 4.7 Flash (Local)",
          "reasoning": false
        }
      ]
    },

```

and then run 

```
pi --model 'glm-4.7-flash:latest'

```

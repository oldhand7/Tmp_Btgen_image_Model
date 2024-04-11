import requests

# Define the GraphQL query and variables as a dictionary
graphql_request = {
    "query": "mutation UpdateImagesGeneration($data: ImageGenerationInput!) { updateImagesGeneration(data: $data) { status }}",
    "variables": {
        "data": {
            "images": [
                {
                    "url": "http://127.0.0.1:8888/files/2024-03-22/68e8fad1-9a14-4c41-85c2-0d6070b3bf97.png",
                    "prompt": "Summer"
                }
            ]
        }
    }
}

# Define the headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer d7093c26-f508-47e4-833c-757706a62bfd",
    "Cookie": "jgb_cs=s%3A96Q5_rfHS3EaRCEV6iKlsX7u_zm4naZD.yKB%2BJ35mmaGGryviAAagXeCrvkyAC9K4rCLjc4Xzd8c"
}

# Define the GraphQL API endpoint
url = "https://stage-graphql.beautifultechnologies.app/"

# Send the HTTP request using the `requests` library
response = requests.post(url, json=graphql_request, headers=headers)

# Print the response content and status code
print(response.content)
print(response.status_code)
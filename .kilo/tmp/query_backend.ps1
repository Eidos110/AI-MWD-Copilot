# Railway GraphQL API - Phase 1: Query Backend Instance Config
$token = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"
$apiUrl = "https://backboard.railway.com/graphql"

$query = @'
query {
  serviceInstance(id: "25d24257-bda3-4fa0-a6ab-4f797107da8e") {
    id
    config
    service { id name }
  }
}
'@

$body = @{ query = $query } | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body -ContentType "application/json" -Headers @{ Authorization = "Bearer $token" }
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    $_.Exception.Response | Format-List *
}

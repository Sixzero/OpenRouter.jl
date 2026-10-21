import MbedTLS
using Base64: base64encode

# Google OAuth access tokens for Vertex AI (no API keys there). Credentials = a service
# account or `authorized_user` (gcloud ADC) JSON: the env var holds its content or a path.
# Tokens live ~1h; cached per credential until 5 minutes before expiry.

const GOOGLE_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
const GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
const _GOOGLE_TOKENS = Dict{UInt,Tuple{String,Float64}}()  # hash(cred) => (token, expires_at); no secrets kept
const _GOOGLE_TOKENS_LOCK = ReentrantLock()

b64url(x) = rstrip(replace(base64encode(x), '+' => '-', '/' => '_'), '=')

"RS256-signed JWT assertion for a service account (RFC 7523 bearer grant)."
function google_jwt(sa::AbstractDict, iat::Integer=round(Int, time()))
    header = b64url("""{"alg":"RS256","typ":"JWT"}""")
    claims = b64url(JSON3.write(Dict("iss" => sa["client_email"], "scope" => GOOGLE_SCOPE,
        "aud" => get(sa, "token_uri", GOOGLE_TOKEN_URI), "iat" => iat, "exp" => iat + 3600)))
    msg = "$header.$claims"
    key = MbedTLS.PKContext()
    MbedTLS.parse_key!(key, sa["private_key"])
    rng = MbedTLS.CtrDrbg()
    MbedTLS.seed!(rng, MbedTLS.Entropy())
    sig = MbedTLS.sign(key, MbedTLS.MD_SHA256, MbedTLS.digest(MbedTLS.MD_SHA256, msg), rng)
    return "$msg.$(b64url(sig))"
end

function fetch_google_token(cred::AbstractDict)
    form = if get(cred, "type", "") == "authorized_user"
        Dict("grant_type" => "refresh_token", "client_id" => cred["client_id"],
             "client_secret" => cred["client_secret"], "refresh_token" => cred["refresh_token"])
    else
        Dict("grant_type" => "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion" => google_jwt(cred))
    end
    resp = HTTP.post(get(cred, "token_uri", GOOGLE_TOKEN_URI),
        ["Content-Type" => "application/x-www-form-urlencoded"], HTTP.URIs.escapeuri(form);
        status_exception=false, readtimeout=30, retry=false)
    resp.status == 200 || error("Google token request failed ($(resp.status)): $(String(resp.body))")
    r = JSON3.read(resp.body)
    return String(r.access_token), time() + get(r, :expires_in, 3600)
end

"Access token for credentials JSON (content or file path), cached until near expiry."
function google_access_token(cred_src::AbstractString)
    lock(_GOOGLE_TOKENS_LOCK) do
        key = hash(cred_src)
        tok, exp = get(_GOOGLE_TOKENS, key, ("", 0.0))
        time() < exp - 300 && return tok
        json = startswith(lstrip(cred_src), "{") ? cred_src : read(cred_src, String)
        tok, exp = fetch_google_token(JSON3.read(json, Dict{String,Any}))
        _GOOGLE_TOKENS[key] = (tok, exp)
        return tok
    end
end

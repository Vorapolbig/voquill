ALTER TABLE stt_providers
  ADD COLUMN cf_access_client_id TEXT NOT NULL DEFAULT '',
  ADD COLUMN cf_access_client_secret_encrypted TEXT NOT NULL DEFAULT '';

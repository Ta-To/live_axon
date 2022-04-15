import Config

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :live_axon, LiveAxonWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "w5NES8EgTOEJiRB6TtLLT0J5HguR/SJlJZVAm+C0GvFcRh6ZP/PO1KOyM6Ut9Dni",
  server: false

# In test we don't send emails.
config :live_axon, LiveAxon.Mailer,
  adapter: Swoosh.Adapters.Test

# Print only warnings and errors during test
config :logger, level: :warn

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

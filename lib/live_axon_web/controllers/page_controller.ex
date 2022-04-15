defmodule LiveAxonWeb.PageController do
  use LiveAxonWeb, :controller

  def index(conn, _params) do
    render(conn, "index.html")
  end
end

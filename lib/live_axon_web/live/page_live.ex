defmodule LiveAxonWeb.PageLive do
  use LiveAxonWeb, :live_view
  require Axon

  @impl true
  def mount(_, _, socket) do
    {:ok, socket |> assign(ans: nil)}
  end

  @impl true
  def handle_event("clear", _params, socket) do
    {:noreply, push_event(socket, "clear", %{})}
  end

  @impl true
  def handle_event("predict", _params, socket) do
    {:noreply, push_event(socket, "predict", %{})}
  end

  @impl true
  def handle_event("predict_axon", %{"data" => data}, socket) do
    ans =
      data
      |> convert_image_data_to_tensor()
      |> convert_mnist_predict_data()
      |> predict()

    {:noreply, assign(socket, :ans, ans)}
  end

  def convert_image_data_to_tensor(data) do
    data
    |> Map.to_list()
    |> Enum.map(fn {k, v} -> {String.to_integer(k), v} end)
    |> Enum.sort()
    |> Enum.map(fn {_k, v} -> v end)
    |> Nx.tensor()
  end

  def convert_mnist_predict_data(pixel) do
    {row} = Nx.shape(pixel)

    pixel
    |> Nx.reshape({div(row, 4), 4})
    |> Nx.slice_along_axis(0, 3, axis: 1)
    |> Nx.mean(axes: [-1])
    |> Nx.round()
    |> Nx.reshape({1, 784})
    |> tap(&(&1 |> Nx.reshape({28, 28}) |> Nx.to_heatmap() |> IO.inspect()))
    |> Nx.divide(255.0)
  end

  def predict(pixel) do
    {:ok, params} = :dets.open_file('weight.dets')
    [{1, {model, weight}}] = :dets.lookup(params, 1)
    Axon.predict(model, weight, pixel) |> Nx.argmax() |> Nx.to_number()
  end
end

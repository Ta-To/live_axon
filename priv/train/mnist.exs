defmodule Mnist do
  # exlaで使うデバイスを指定 rcomはamd系 gpuがなかったら最後のhost(cpu）になる
  EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])

  # 必須
  require Axon

  # mnistの画像データをnxで使えるように変換
  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
    # Test split
    |> Enum.split(1750)
  end

  # mnistのラベルデータをnxで使えるように変換
  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
    # Test split
    |> Enum.split(1750)
  end

  # ニューラルネットワークの構築
  # input_shape 28x28
  # dense1 hidden_layer 128
  # dense2 output 10
  defp build_model(input_shape) do
    Axon.input(input_shape)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp train_model(model, train_images, train_labels, epochs) do
    model
    # 損失関数と活性化関数を指定
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
    # 1epochごとに計算する指標を正解率に指定
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), epochs: epochs, compiler: EXLA)
  end

  defp test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator(model_state)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), compiler: EXLA)
  end

  def run do
    # scidataから画像とラベルのバイナリデータをダウンロード
    {images, labels} = Scidata.MNIST.download()

    # Nx用に変換して分割
    {train_images, test_images} = transform_images(images)
    {train_labels, test_labels} = transform_labels(labels)

    # モデルを構築してネットワーク構成を表示
    model = build_model({nil, 784}) |> IO.inspect()

    IO.write("\n\n Training Model \n\n")

    # epoch５で学習を実行し最終的なパラメーターを受け取る
    model_state =
      model
      |> train_model(train_images, train_labels, 5)


    IO.write("\n\n Testing Model \n\n")
    # テストデータで検証
    model
    |> test_model(model_state, test_images, test_labels)
    |> IO.inspect()
    IO.write("\n\n")

    {model, model_state}
  end

  # 上記を実行して detsにモデルと学習済み重みを保存
  def generate_trained_network do
    {model, weight} = Mnist.run()
    :dets.open_file('weight', type: :bag, file: 'weight.dets')
    :dets.insert('weight',{1,{model,weight}})
    :dets.sync('weight')
    :dets.stop
  end
end

Mnist.generate_trained_network()

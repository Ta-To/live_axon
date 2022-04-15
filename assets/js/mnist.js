const Mnist = {
  mounted() {
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    let clickFlg = 0;
    canvas.width = 300
    canvas.height = 300

    const draw = (x, y) => {
      ctx.lineWidth = 10;
      ctx.strokeStyle = 'rgba(255, 255, 255, 1)';
      if (clickFlg == "1") {
        clickFlg = "2";
        ctx.beginPath();
        ctx.lineCap = "round";
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    const setBgColor = () => {
      ctx.fillRect(0,0,300,300);
    }
    setBgColor();

    canvas.addEventListener("mousedown",() => {
      clickFlg = 1;
    })
    canvas.addEventListener("mouseup", () => {
      clickFlg = 0;
    })
    canvas.addEventListener("mousemove",(e) => {
      if(!clickFlg) return false;
      draw(e.offsetX, e.offsetY);
    });

    this.handleEvent("clear", () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setBgColor();
    });

    let canvas2 = document.getElementById("canvas2");
    let ctx2 = canvas2.getContext("2d");
    canvas2.width = 28;
    canvas2.height = 28;

    this.handleEvent("predict", () => {
      ctx2.drawImage(canvas, 0, 0, 300, 300, 0, 0, 28, 28);
      let data = ctx2.getImageData(0, 0, 28, 28);
      ctx2.fillRect(0, 0, 28, 28);
      this.pushEvent("predict_axon", data);
    });
  }
}
export default Mnist;

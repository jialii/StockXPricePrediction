setTimeout(() => {
  console.log("Pricer");
  var bttn = document.createElement("button");
  bttn.setAttribute(
    "onclick",
    `window.location.href = 'http://127.0.0.1:8050/prediction/${
      window.location.href.split("/")[3]
    }'`
  );
  bttn.innerText = "Price Prediction";
  bttn.className = "chakra-button css-2yrtpe";
  document.querySelectorAll(".chakra-button__group")[1].style.width = "461px";
  document.querySelectorAll(".chakra-button__group")[1].appendChild(bttn);
}, 3000);

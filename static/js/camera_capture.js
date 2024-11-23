// 獲取 DOM 元素
const video = document.getElementById('real-time-video');
const realTimeCanvas = document.getElementById('real-time-canvas'); // 新增一個 canvas 元素取代 video
const binaryCanvas = document.getElementById('binary-video');
const captureButton = document.getElementById('capture');
const capturedOriginal = document.getElementById('captured-original');
const capturedBinary = document.getElementById('captured-binary');

// 啟用客戶端攝像頭
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error("無法啟用攝像頭: ", err);
        alert("請允許瀏覽器使用攝像頭以顯示畫面！");
    });

    function dataURItoBlob(dataURI) {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
}

// 傳送影像到伺服器進行處理並接收結果
function processFrame(callback) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg');

    // 發送到伺服器進行處理
    fetch('/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                console.error("伺服器錯誤:", data.error);
                return;
            }

            // 繪製處理後的原始影像到 canvas
            const realTimeContext = realTimeCanvas.getContext('2d');
            const realTimeImage = new Image();
            realTimeImage.onload = () => {
                realTimeContext.clearRect(0, 0, realTimeCanvas.width, realTimeCanvas.height);
                realTimeContext.drawImage(realTimeImage, 0, 0, realTimeCanvas.width, realTimeCanvas.height);
            };
            realTimeImage.src = data.original_image;

            // 將二值化影像繪製到 canvas
            const binaryContext = binaryCanvas.getContext('2d');
            const binaryImage = new Image();
            binaryImage.onload = () => {
                binaryContext.clearRect(0, 0, binaryCanvas.width, binaryCanvas.height);
                binaryContext.drawImage(binaryImage, 0, 0, binaryCanvas.width, binaryCanvas.height);
            };
            binaryImage.src = data.binary_image;

            // 如果有回呼函數，執行回呼函數
            if (callback) callback(data);
        })
        .catch((err) => {
            console.error("傳輸錯誤:", err);
        });
}

// 定時處理即時影像
setInterval(() => {
    processFrame();
}, 50); // 每 100 毫秒傳送一次影像

// 捕捉影像按鈕功能
captureButton.addEventListener('click', () => {
    processFrame((data) => {
        // 更新捕捉影像結果
        capturedOriginal.src = data.original_image;
        capturedBinary.src = data.binary_image;
    });
});

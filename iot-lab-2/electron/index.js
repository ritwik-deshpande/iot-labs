document.onkeydown = updateKey;
document.onkeyup = resetKey;
connect();

var server_port = 65432;
var server_addr = "127.0.0.1";   // the IP address of your Raspberry PI
var isConnected = false;

const net = require('net');
var client;

if(isConnected)
{
    client.on('data', (data) => {
        document.getElementById("bluetooth").innerHTML = data;
        console.log(data.toString());
        // client.end();
        // client.destroy();
    });
}

function establishConnection()
{
    client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        isConnected = true;
        console.log('connected to server!');
    });
}

function sendMessage(){
    var input = document.getElementById("message").value;

    client.write(`${input}\r\n`);

    client.on('end', () => {
        console.log('disconnected from server');
    });


}

function sendDirection(direction)
{
    if(isConnected)
    {
        client.write(`${direction}\r\n`);
    }
    else
    console.log("Not connected");
}

// for detecting which key is been pressed w,a,s,d
function updateKey(e) {

    e = e || window.event;

    if (e.keyCode == '87') {
        // up (w)
        document.getElementById("upArrow").style.color = "green";
        sendDirection("87");
    }
    else if (e.keyCode == '83') {
        // down (s)
        document.getElementById("downArrow").style.color = "green";
        sendDirection("83");
    }
    else if (e.keyCode == '65') {
        // left (a)
        document.getElementById("leftArrow").style.color = "green";
        sendDirection("65");
    }
    else if (e.keyCode == '68') {
        // right (d)
        document.getElementById("rightArrow").style.color = "green";
        sendDirection("68");
    }
}

// reset the key to the start state 
function resetKey(e) {

    e = e || window.event;

    document.getElementById("upArrow").style.color = "grey";
    document.getElementById("downArrow").style.color = "grey";
    document.getElementById("leftArrow").style.color = "grey";
    document.getElementById("rightArrow").style.color = "grey";
}


// update data for every 50ms
function update_data(){
    sendMessage();
}

function connect()
{
    setInterval(function(){
        // get image from python server
        if(!isConnected)
        establishConnection();
    }, 5000)
}

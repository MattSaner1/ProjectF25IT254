const int buttonPin = 3;
const int ledPin = 13;


bool lastButtonState = LOW;
bool recording = false;


void setup() {
  pinMode(buttonPin, INPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}


void loop() {
  bool buttonState = digitalRead(buttonPin);


  if (buttonState == HIGH && lastButtonState == LOW) {
    delay(150); // debounce
    if (!recording) {
      recording = true;
      digitalWrite(ledPin, HIGH);
      Serial.println("START");
      Serial.flush();
    } else {
      recording = false;
      digitalWrite(ledPin, LOW);
      Serial.println("STOP");
      Serial.flush();
    }
  }


  lastButtonState = buttonState;
}


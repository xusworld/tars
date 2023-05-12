FLATC=../../third-party/flatbuffers/bin/flatc
$FLATC -c -b --gen-object-api --reflect-names ./proto/*.fbs

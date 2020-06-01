# Bundle together the code in one file so easily can include it in Colaboratory without having
# to install through pip

mkdir /tmp/apa
cp captchagenerator/*.py /tmp/apa/
rm /tmp/apa/__init__.py

echo "# GENERATED DO NOT EDIT!" > /tmp/whatever.py
cat /tmp/apa/*.py >> /tmp/whatever.py

grep -v "import captchagenerator." /tmp/whatever.py > /tmp/tocolab.py

sed -i.bak 's/capgenerator\.//g' /tmp/tocolab.py
sed -i.bak 's/imagemanip\.//g' /tmp/tocolab.py
sed -i.bak 's/classify\.//g' /tmp/tocolab.py
sed -i.bak 's/PILasOPENCVFix\.//g' /tmp/tocolab.py


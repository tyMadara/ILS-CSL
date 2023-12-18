# initialize the python environment
pip install --upgrade pip
pip install -r requirements.txt

# download and setup BI-CaMML
if [ ! -d "BI-CaMML" ]; then
    wget https://bayesian-intelligence.com/software/BI-CaMML-1.4.2.zip
    unzip BI-CaMML-1.4.2.zip
    rm BI-CaMML-1.4.2.zip
fi
mkdir BI-CaMML/anc_file BI-CaMML/BN_record BI-CaMML/out_BNs out/prior-iter
chmod 775 BI-CaMML/camml.sh

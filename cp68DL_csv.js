function getData(idNames, category) {

  function downloadURI(uri) {
    var link = document.createElement("a");
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    delete link;
  };

  const prefix = {
    metastock: "http://www.cophieu68.vn/export/metastock.php?id=",
    excelfull: "http://www.cophieu68.vn/export/excelfull.php?id=",
    reportfinance: "http://www.cophieu68.vn/export/reportfinance.php?id=",
    indexfinance: "http://www.cophieu68.vn/export/indexfinance.php?id=",
    events: "http://www.cophieu68.vn/export/events.php?id="
  }

  let links = {
    metastock: [],
    excelfull: [],
    reportfinance: [],
    indexfinance: [],
    events: []
  };


  for (let i = 0; i < idNames.length; i++) {
    links[category].push(prefix[category] + idNames[i]);
  }

  let counter = 0;
  let i = setInterval(function () {
    downloadURI(links[category][counter]);
    counter++;
    limitDownload = links.length - 2;
    if (counter === limitDownload) {
      clearInterval(i);
    }
  }, 3000);

};

let idNames = ['ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
  		  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
  		  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC',
		  "^VNINDEX", "^HASTC", "^UPCOM", 
			'ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
			  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
			  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC', 
			  "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
			  "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
		   "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
		   "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
		   "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
		   "HSG", "HDG", "HCM", "HPG", "HBC", 
		   "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
		   "MBB", "MSN", "MWG", 
			"NKG", "NLG", "NT2", "NVL", "NBB",
			"PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
			"QCG", "REE",  
			"SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
				"VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS",
				"CVT", "C32", "SMC", "NTC", "SWC", "SBC", "VE9", "KLF", "HVA", "VHG", "HKB",
			   "ASA", "KDM", "NVT", "MBS", "HDB", "VND", "SBS", "DVN"]
		   
let category = "excelfull";

getData(idNames, category)
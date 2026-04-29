OPENQASM 2.0;
include "qelib1.inc";

qreg q[55];
creg rec[217];

reset q[0];
reset q[1];
reset q[2];
reset q[3];
reset q[4];
reset q[5];
reset q[6];
reset q[7];
reset q[8];
reset q[9];
reset q[10];
reset q[11];
reset q[12];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
reset q[18];
reset q[19];
reset q[20];
reset q[21];
reset q[22];
reset q[23];
reset q[24];
reset q[25];
reset q[26];
reset q[27];
reset q[28];
reset q[29];
reset q[30];
reset q[31];
reset q[32];
reset q[33];
reset q[34];
reset q[35];
reset q[36];
reset q[47];
reset q[37];
reset q[49];
reset q[48];
reset q[38];
reset q[39];
reset q[40];
reset q[50];
reset q[51];
reset q[52];
reset q[41];
reset q[42];
reset q[43];
reset q[44];
reset q[53];
reset q[54];
reset q[45];
reset q[46];
barrier q;

reset q[38]; h q[38]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[53]; h q[53]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[41]; h q[41]; // decomposed RX
reset q[50]; h q[50]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[46]; h q[46]; // decomposed RX
reset q[54]; h q[54]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
reset q[47]; h q[47]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[48]; h q[48]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[49]; h q[49]; // decomposed RX
barrier q;

cx q[37], q[0];
cx q[41], q[8];
cx q[50], q[26];
cx q[39], q[3];
cx q[46], q[17];
cx q[54], q[33];
cx q[42], q[9];
cx q[51], q[27];
cx q[40], q[4];
cx q[47], q[18];
cx q[43], q[10];
cx q[52], q[28];
cx q[48], q[19];
cx q[44], q[11];
cx q[49], q[20];
barrier q;

cx q[37], q[1];
cx q[41], q[12];
cx q[50], q[30];
cx q[39], q[5];
cx q[46], q[22];
cx q[54], q[35];
cx q[42], q[13];
cx q[51], q[31];
cx q[40], q[6];
cx q[47], q[23];
cx q[43], q[14];
cx q[52], q[32];
cx q[48], q[24];
cx q[44], q[15];
cx q[49], q[25];
barrier q;

cx q[38], q[1];
cx q[45], q[12];
cx q[53], q[30];
cx q[41], q[5];
cx q[50], q[22];
cx q[39], q[2];
cx q[46], q[13];
cx q[54], q[31];
cx q[42], q[6];
cx q[51], q[23];
cx q[47], q[14];
cx q[43], q[7];
cx q[52], q[24];
cx q[48], q[15];
cx q[49], q[16];
barrier q;

cx q[38], q[8];
cx q[45], q[26];
cx q[53], q[36];
cx q[37], q[3];
cx q[41], q[17];
cx q[50], q[33];
cx q[39], q[9];
cx q[46], q[27];
cx q[42], q[18];
cx q[51], q[34];
cx q[40], q[10];
cx q[47], q[28];
cx q[43], q[19];
cx q[48], q[29];
cx q[44], q[20];
barrier q;

cx q[38], q[3];
cx q[45], q[17];
cx q[53], q[33];
cx q[41], q[9];
cx q[50], q[27];
cx q[39], q[4];
cx q[46], q[18];
cx q[54], q[34];
cx q[42], q[10];
cx q[51], q[28];
cx q[47], q[19];
cx q[43], q[11];
cx q[52], q[29];
cx q[48], q[20];
cx q[49], q[21];
barrier q;

cx q[38], q[5];
cx q[45], q[22];
cx q[53], q[35];
cx q[37], q[2];
cx q[41], q[13];
cx q[50], q[31];
cx q[39], q[6];
cx q[46], q[23];
cx q[42], q[14];
cx q[51], q[32];
cx q[40], q[7];
cx q[47], q[24];
cx q[43], q[15];
cx q[48], q[25];
cx q[44], q[16];
barrier q;

h q[38]; measure q[38] -> rec[0]; h q[38]; // decomposed MX
h q[45]; measure q[45] -> rec[1]; h q[45]; // decomposed MX
h q[53]; measure q[53] -> rec[2]; h q[53]; // decomposed MX
h q[37]; measure q[37] -> rec[3]; h q[37]; // decomposed MX
h q[41]; measure q[41] -> rec[4]; h q[41]; // decomposed MX
h q[50]; measure q[50] -> rec[5]; h q[50]; // decomposed MX
h q[39]; measure q[39] -> rec[6]; h q[39]; // decomposed MX
h q[46]; measure q[46] -> rec[7]; h q[46]; // decomposed MX
h q[54]; measure q[54] -> rec[8]; h q[54]; // decomposed MX
h q[42]; measure q[42] -> rec[9]; h q[42]; // decomposed MX
h q[51]; measure q[51] -> rec[10]; h q[51]; // decomposed MX
h q[40]; measure q[40] -> rec[11]; h q[40]; // decomposed MX
h q[47]; measure q[47] -> rec[12]; h q[47]; // decomposed MX
h q[43]; measure q[43] -> rec[13]; h q[43]; // decomposed MX
h q[52]; measure q[52] -> rec[14]; h q[52]; // decomposed MX
h q[48]; measure q[48] -> rec[15]; h q[48]; // decomposed MX
h q[44]; measure q[44] -> rec[16]; h q[44]; // decomposed MX
h q[49]; measure q[49] -> rec[17]; h q[49]; // decomposed MX
reset q[38];
reset q[45];
reset q[53];
reset q[37];
reset q[41];
reset q[50];
reset q[39];
reset q[46];
reset q[54];
reset q[42];
reset q[51];
reset q[40];
reset q[47];
reset q[43];
reset q[52];
reset q[48];
reset q[44];
reset q[49];
barrier q;

cx q[0], q[37];
cx q[8], q[41];
cx q[26], q[50];
cx q[3], q[39];
cx q[17], q[46];
cx q[33], q[54];
cx q[9], q[42];
cx q[27], q[51];
cx q[4], q[40];
cx q[18], q[47];
cx q[10], q[43];
cx q[28], q[52];
cx q[19], q[48];
cx q[11], q[44];
cx q[20], q[49];
barrier q;

cx q[1], q[37];
cx q[12], q[41];
cx q[30], q[50];
cx q[5], q[39];
cx q[22], q[46];
cx q[35], q[54];
cx q[13], q[42];
cx q[31], q[51];
cx q[6], q[40];
cx q[23], q[47];
cx q[14], q[43];
cx q[32], q[52];
cx q[24], q[48];
cx q[15], q[44];
cx q[25], q[49];
barrier q;

cx q[1], q[38];
cx q[12], q[45];
cx q[30], q[53];
cx q[5], q[41];
cx q[22], q[50];
cx q[2], q[39];
cx q[13], q[46];
cx q[31], q[54];
cx q[6], q[42];
cx q[23], q[51];
cx q[14], q[47];
cx q[7], q[43];
cx q[24], q[52];
cx q[15], q[48];
cx q[16], q[49];
barrier q;

cx q[8], q[38];
cx q[26], q[45];
cx q[36], q[53];
cx q[3], q[37];
cx q[17], q[41];
cx q[33], q[50];
cx q[9], q[39];
cx q[27], q[46];
cx q[18], q[42];
cx q[34], q[51];
cx q[10], q[40];
cx q[28], q[47];
cx q[19], q[43];
cx q[29], q[48];
cx q[20], q[44];
barrier q;

cx q[3], q[38];
cx q[17], q[45];
cx q[33], q[53];
cx q[9], q[41];
cx q[27], q[50];
cx q[4], q[39];
cx q[18], q[46];
cx q[34], q[54];
cx q[10], q[42];
cx q[28], q[51];
cx q[19], q[47];
cx q[11], q[43];
cx q[29], q[52];
cx q[20], q[48];
cx q[21], q[49];
barrier q;

cx q[5], q[38];
cx q[22], q[45];
cx q[35], q[53];
cx q[2], q[37];
cx q[13], q[41];
cx q[31], q[50];
cx q[6], q[39];
cx q[23], q[46];
cx q[14], q[42];
cx q[32], q[51];
cx q[7], q[40];
cx q[24], q[47];
cx q[15], q[43];
cx q[25], q[48];
cx q[16], q[44];
barrier q;

measure q[38] -> rec[18];
measure q[45] -> rec[19];
measure q[53] -> rec[20];
measure q[37] -> rec[21];
measure q[41] -> rec[22];
measure q[50] -> rec[23];
measure q[39] -> rec[24];
measure q[46] -> rec[25];
measure q[54] -> rec[26];
measure q[42] -> rec[27];
measure q[51] -> rec[28];
measure q[40] -> rec[29];
measure q[47] -> rec[30];
measure q[43] -> rec[31];
measure q[52] -> rec[32];
measure q[48] -> rec[33];
measure q[44] -> rec[34];
measure q[49] -> rec[35];
barrier q;

reset q[38]; h q[38]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[53]; h q[53]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[41]; h q[41]; // decomposed RX
reset q[50]; h q[50]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[46]; h q[46]; // decomposed RX
reset q[54]; h q[54]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
reset q[47]; h q[47]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[48]; h q[48]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[49]; h q[49]; // decomposed RX
barrier q;

cx q[37], q[0];
cx q[41], q[8];
cx q[50], q[26];
cx q[39], q[3];
cx q[46], q[17];
cx q[54], q[33];
cx q[42], q[9];
cx q[51], q[27];
cx q[40], q[4];
cx q[47], q[18];
cx q[43], q[10];
cx q[52], q[28];
cx q[48], q[19];
cx q[44], q[11];
cx q[49], q[20];
barrier q;

cx q[37], q[1];
cx q[41], q[12];
cx q[50], q[30];
cx q[39], q[5];
cx q[46], q[22];
cx q[54], q[35];
cx q[42], q[13];
cx q[51], q[31];
cx q[40], q[6];
cx q[47], q[23];
cx q[43], q[14];
cx q[52], q[32];
cx q[48], q[24];
cx q[44], q[15];
cx q[49], q[25];
barrier q;

cx q[38], q[1];
cx q[45], q[12];
cx q[53], q[30];
cx q[41], q[5];
cx q[50], q[22];
cx q[39], q[2];
cx q[46], q[13];
cx q[54], q[31];
cx q[42], q[6];
cx q[51], q[23];
cx q[47], q[14];
cx q[43], q[7];
cx q[52], q[24];
cx q[48], q[15];
cx q[49], q[16];
barrier q;

cx q[38], q[8];
cx q[45], q[26];
cx q[53], q[36];
cx q[37], q[3];
cx q[41], q[17];
cx q[50], q[33];
cx q[39], q[9];
cx q[46], q[27];
cx q[42], q[18];
cx q[51], q[34];
cx q[40], q[10];
cx q[47], q[28];
cx q[43], q[19];
cx q[48], q[29];
cx q[44], q[20];
barrier q;

cx q[38], q[3];
cx q[45], q[17];
cx q[53], q[33];
cx q[41], q[9];
cx q[50], q[27];
cx q[39], q[4];
cx q[46], q[18];
cx q[54], q[34];
cx q[42], q[10];
cx q[51], q[28];
cx q[47], q[19];
cx q[43], q[11];
cx q[52], q[29];
cx q[48], q[20];
cx q[49], q[21];
barrier q;

cx q[38], q[5];
cx q[45], q[22];
cx q[53], q[35];
cx q[37], q[2];
cx q[41], q[13];
cx q[50], q[31];
cx q[39], q[6];
cx q[46], q[23];
cx q[42], q[14];
cx q[51], q[32];
cx q[40], q[7];
cx q[47], q[24];
cx q[43], q[15];
cx q[48], q[25];
cx q[44], q[16];
barrier q;

h q[38]; measure q[38] -> rec[36]; h q[38]; // decomposed MX
h q[45]; measure q[45] -> rec[37]; h q[45]; // decomposed MX
h q[53]; measure q[53] -> rec[38]; h q[53]; // decomposed MX
h q[37]; measure q[37] -> rec[39]; h q[37]; // decomposed MX
h q[41]; measure q[41] -> rec[40]; h q[41]; // decomposed MX
h q[50]; measure q[50] -> rec[41]; h q[50]; // decomposed MX
h q[39]; measure q[39] -> rec[42]; h q[39]; // decomposed MX
h q[46]; measure q[46] -> rec[43]; h q[46]; // decomposed MX
h q[54]; measure q[54] -> rec[44]; h q[54]; // decomposed MX
h q[42]; measure q[42] -> rec[45]; h q[42]; // decomposed MX
h q[51]; measure q[51] -> rec[46]; h q[51]; // decomposed MX
h q[40]; measure q[40] -> rec[47]; h q[40]; // decomposed MX
h q[47]; measure q[47] -> rec[48]; h q[47]; // decomposed MX
h q[43]; measure q[43] -> rec[49]; h q[43]; // decomposed MX
h q[52]; measure q[52] -> rec[50]; h q[52]; // decomposed MX
h q[48]; measure q[48] -> rec[51]; h q[48]; // decomposed MX
h q[44]; measure q[44] -> rec[52]; h q[44]; // decomposed MX
h q[49]; measure q[49] -> rec[53]; h q[49]; // decomposed MX
barrier q;

reset q[38];
reset q[45];
reset q[53];
reset q[37];
reset q[41];
reset q[50];
reset q[39];
reset q[46];
reset q[54];
reset q[42];
reset q[51];
reset q[40];
reset q[47];
reset q[43];
reset q[52];
reset q[48];
reset q[44];
reset q[49];
barrier q;

cx q[0], q[37];
cx q[8], q[41];
cx q[26], q[50];
cx q[3], q[39];
cx q[17], q[46];
cx q[33], q[54];
cx q[9], q[42];
cx q[27], q[51];
cx q[4], q[40];
cx q[18], q[47];
cx q[10], q[43];
cx q[28], q[52];
cx q[19], q[48];
cx q[11], q[44];
cx q[20], q[49];
barrier q;

cx q[1], q[37];
cx q[12], q[41];
cx q[30], q[50];
cx q[5], q[39];
cx q[22], q[46];
cx q[35], q[54];
cx q[13], q[42];
cx q[31], q[51];
cx q[6], q[40];
cx q[23], q[47];
cx q[14], q[43];
cx q[32], q[52];
cx q[24], q[48];
cx q[15], q[44];
cx q[25], q[49];
barrier q;

cx q[1], q[38];
cx q[12], q[45];
cx q[30], q[53];
cx q[5], q[41];
cx q[22], q[50];
cx q[2], q[39];
cx q[13], q[46];
cx q[31], q[54];
cx q[6], q[42];
cx q[23], q[51];
cx q[14], q[47];
cx q[7], q[43];
cx q[24], q[52];
cx q[15], q[48];
cx q[16], q[49];
barrier q;

cx q[8], q[38];
cx q[26], q[45];
cx q[36], q[53];
cx q[3], q[37];
cx q[17], q[41];
cx q[33], q[50];
cx q[9], q[39];
cx q[27], q[46];
cx q[18], q[42];
cx q[34], q[51];
cx q[10], q[40];
cx q[28], q[47];
cx q[19], q[43];
cx q[29], q[48];
cx q[20], q[44];
barrier q;

cx q[3], q[38];
cx q[17], q[45];
cx q[33], q[53];
cx q[9], q[41];
cx q[27], q[50];
cx q[4], q[39];
cx q[18], q[46];
cx q[34], q[54];
cx q[10], q[42];
cx q[28], q[51];
cx q[19], q[47];
cx q[11], q[43];
cx q[29], q[52];
cx q[20], q[48];
cx q[21], q[49];
barrier q;

cx q[5], q[38];
cx q[22], q[45];
cx q[35], q[53];
cx q[2], q[37];
cx q[13], q[41];
cx q[31], q[50];
cx q[6], q[39];
cx q[23], q[46];
cx q[14], q[42];
cx q[32], q[51];
cx q[7], q[40];
cx q[24], q[47];
cx q[15], q[43];
cx q[25], q[48];
cx q[16], q[44];
barrier q;

measure q[38] -> rec[54];
measure q[45] -> rec[55];
measure q[53] -> rec[56];
measure q[37] -> rec[57];
measure q[41] -> rec[58];
measure q[50] -> rec[59];
measure q[39] -> rec[60];
measure q[46] -> rec[61];
measure q[54] -> rec[62];
measure q[42] -> rec[63];
measure q[51] -> rec[64];
measure q[40] -> rec[65];
measure q[47] -> rec[66];
measure q[43] -> rec[67];
measure q[52] -> rec[68];
measure q[48] -> rec[69];
measure q[44] -> rec[70];
measure q[49] -> rec[71];
barrier q;

reset q[38]; h q[38]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[53]; h q[53]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[41]; h q[41]; // decomposed RX
reset q[50]; h q[50]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[46]; h q[46]; // decomposed RX
reset q[54]; h q[54]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
reset q[47]; h q[47]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[48]; h q[48]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[49]; h q[49]; // decomposed RX
barrier q;

cx q[37], q[0];
cx q[41], q[8];
cx q[50], q[26];
cx q[39], q[3];
cx q[46], q[17];
cx q[54], q[33];
cx q[42], q[9];
cx q[51], q[27];
cx q[40], q[4];
cx q[47], q[18];
cx q[43], q[10];
cx q[52], q[28];
cx q[48], q[19];
cx q[44], q[11];
cx q[49], q[20];
barrier q;

cx q[37], q[1];
cx q[41], q[12];
cx q[50], q[30];
cx q[39], q[5];
cx q[46], q[22];
cx q[54], q[35];
cx q[42], q[13];
cx q[51], q[31];
cx q[40], q[6];
cx q[47], q[23];
cx q[43], q[14];
cx q[52], q[32];
cx q[48], q[24];
cx q[44], q[15];
cx q[49], q[25];
barrier q;

cx q[38], q[1];
cx q[45], q[12];
cx q[53], q[30];
cx q[41], q[5];
cx q[50], q[22];
cx q[39], q[2];
cx q[46], q[13];
cx q[54], q[31];
cx q[42], q[6];
cx q[51], q[23];
cx q[47], q[14];
cx q[43], q[7];
cx q[52], q[24];
cx q[48], q[15];
cx q[49], q[16];
barrier q;

cx q[38], q[8];
cx q[45], q[26];
cx q[53], q[36];
cx q[37], q[3];
cx q[41], q[17];
cx q[50], q[33];
cx q[39], q[9];
cx q[46], q[27];
cx q[42], q[18];
cx q[51], q[34];
cx q[40], q[10];
cx q[47], q[28];
cx q[43], q[19];
cx q[48], q[29];
cx q[44], q[20];
barrier q;

cx q[38], q[3];
cx q[45], q[17];
cx q[53], q[33];
cx q[41], q[9];
cx q[50], q[27];
cx q[39], q[4];
cx q[46], q[18];
cx q[54], q[34];
cx q[42], q[10];
cx q[51], q[28];
cx q[47], q[19];
cx q[43], q[11];
cx q[52], q[29];
cx q[48], q[20];
cx q[49], q[21];
barrier q;

cx q[38], q[5];
cx q[45], q[22];
cx q[53], q[35];
cx q[37], q[2];
cx q[41], q[13];
cx q[50], q[31];
cx q[39], q[6];
cx q[46], q[23];
cx q[42], q[14];
cx q[51], q[32];
cx q[40], q[7];
cx q[47], q[24];
cx q[43], q[15];
cx q[48], q[25];
cx q[44], q[16];
barrier q;

h q[38]; measure q[38] -> rec[72]; h q[38]; // decomposed MX
h q[45]; measure q[45] -> rec[73]; h q[45]; // decomposed MX
h q[53]; measure q[53] -> rec[74]; h q[53]; // decomposed MX
h q[37]; measure q[37] -> rec[75]; h q[37]; // decomposed MX
h q[41]; measure q[41] -> rec[76]; h q[41]; // decomposed MX
h q[50]; measure q[50] -> rec[77]; h q[50]; // decomposed MX
h q[39]; measure q[39] -> rec[78]; h q[39]; // decomposed MX
h q[46]; measure q[46] -> rec[79]; h q[46]; // decomposed MX
h q[54]; measure q[54] -> rec[80]; h q[54]; // decomposed MX
h q[42]; measure q[42] -> rec[81]; h q[42]; // decomposed MX
h q[51]; measure q[51] -> rec[82]; h q[51]; // decomposed MX
h q[40]; measure q[40] -> rec[83]; h q[40]; // decomposed MX
h q[47]; measure q[47] -> rec[84]; h q[47]; // decomposed MX
h q[43]; measure q[43] -> rec[85]; h q[43]; // decomposed MX
h q[52]; measure q[52] -> rec[86]; h q[52]; // decomposed MX
h q[48]; measure q[48] -> rec[87]; h q[48]; // decomposed MX
h q[44]; measure q[44] -> rec[88]; h q[44]; // decomposed MX
h q[49]; measure q[49] -> rec[89]; h q[49]; // decomposed MX
barrier q;

reset q[38];
reset q[45];
reset q[53];
reset q[37];
reset q[41];
reset q[50];
reset q[39];
reset q[46];
reset q[54];
reset q[42];
reset q[51];
reset q[40];
reset q[47];
reset q[43];
reset q[52];
reset q[48];
reset q[44];
reset q[49];
barrier q;

cx q[0], q[37];
cx q[8], q[41];
cx q[26], q[50];
cx q[3], q[39];
cx q[17], q[46];
cx q[33], q[54];
cx q[9], q[42];
cx q[27], q[51];
cx q[4], q[40];
cx q[18], q[47];
cx q[10], q[43];
cx q[28], q[52];
cx q[19], q[48];
cx q[11], q[44];
cx q[20], q[49];
barrier q;

cx q[1], q[37];
cx q[12], q[41];
cx q[30], q[50];
cx q[5], q[39];
cx q[22], q[46];
cx q[35], q[54];
cx q[13], q[42];
cx q[31], q[51];
cx q[6], q[40];
cx q[23], q[47];
cx q[14], q[43];
cx q[32], q[52];
cx q[24], q[48];
cx q[15], q[44];
cx q[25], q[49];
barrier q;

cx q[1], q[38];
cx q[12], q[45];
cx q[30], q[53];
cx q[5], q[41];
cx q[22], q[50];
cx q[2], q[39];
cx q[13], q[46];
cx q[31], q[54];
cx q[6], q[42];
cx q[23], q[51];
cx q[14], q[47];
cx q[7], q[43];
cx q[24], q[52];
cx q[15], q[48];
cx q[16], q[49];
barrier q;

cx q[8], q[38];
cx q[26], q[45];
cx q[36], q[53];
cx q[3], q[37];
cx q[17], q[41];
cx q[33], q[50];
cx q[9], q[39];
cx q[27], q[46];
cx q[18], q[42];
cx q[34], q[51];
cx q[10], q[40];
cx q[28], q[47];
cx q[19], q[43];
cx q[29], q[48];
cx q[20], q[44];
barrier q;

cx q[3], q[38];
cx q[17], q[45];
cx q[33], q[53];
cx q[9], q[41];
cx q[27], q[50];
cx q[4], q[39];
cx q[18], q[46];
cx q[34], q[54];
cx q[10], q[42];
cx q[28], q[51];
cx q[19], q[47];
cx q[11], q[43];
cx q[29], q[52];
cx q[20], q[48];
cx q[21], q[49];
barrier q;

cx q[5], q[38];
cx q[22], q[45];
cx q[35], q[53];
cx q[2], q[37];
cx q[13], q[41];
cx q[31], q[50];
cx q[6], q[39];
cx q[23], q[46];
cx q[14], q[42];
cx q[32], q[51];
cx q[7], q[40];
cx q[24], q[47];
cx q[15], q[43];
cx q[25], q[48];
cx q[16], q[44];
barrier q;

measure q[38] -> rec[90];
measure q[45] -> rec[91];
measure q[53] -> rec[92];
measure q[37] -> rec[93];
measure q[41] -> rec[94];
measure q[50] -> rec[95];
measure q[39] -> rec[96];
measure q[46] -> rec[97];
measure q[54] -> rec[98];
measure q[42] -> rec[99];
measure q[51] -> rec[100];
measure q[40] -> rec[101];
measure q[47] -> rec[102];
measure q[43] -> rec[103];
measure q[52] -> rec[104];
measure q[48] -> rec[105];
measure q[44] -> rec[106];
measure q[49] -> rec[107];
barrier q;

reset q[38]; h q[38]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[53]; h q[53]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[41]; h q[41]; // decomposed RX
reset q[50]; h q[50]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[46]; h q[46]; // decomposed RX
reset q[54]; h q[54]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
reset q[47]; h q[47]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[48]; h q[48]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[49]; h q[49]; // decomposed RX
barrier q;

cx q[37], q[0];
cx q[41], q[8];
cx q[50], q[26];
cx q[39], q[3];
cx q[46], q[17];
cx q[54], q[33];
cx q[42], q[9];
cx q[51], q[27];
cx q[40], q[4];
cx q[47], q[18];
cx q[43], q[10];
cx q[52], q[28];
cx q[48], q[19];
cx q[44], q[11];
cx q[49], q[20];
barrier q;

cx q[37], q[1];
cx q[41], q[12];
cx q[50], q[30];
cx q[39], q[5];
cx q[46], q[22];
cx q[54], q[35];
cx q[42], q[13];
cx q[51], q[31];
cx q[40], q[6];
cx q[47], q[23];
cx q[43], q[14];
cx q[52], q[32];
cx q[48], q[24];
cx q[44], q[15];
cx q[49], q[25];
barrier q;

cx q[38], q[1];
cx q[45], q[12];
cx q[53], q[30];
cx q[41], q[5];
cx q[50], q[22];
cx q[39], q[2];
cx q[46], q[13];
cx q[54], q[31];
cx q[42], q[6];
cx q[51], q[23];
cx q[47], q[14];
cx q[43], q[7];
cx q[52], q[24];
cx q[48], q[15];
cx q[49], q[16];
barrier q;

cx q[38], q[8];
cx q[45], q[26];
cx q[53], q[36];
cx q[37], q[3];
cx q[41], q[17];
cx q[50], q[33];
cx q[39], q[9];
cx q[46], q[27];
cx q[42], q[18];
cx q[51], q[34];
cx q[40], q[10];
cx q[47], q[28];
cx q[43], q[19];
cx q[48], q[29];
cx q[44], q[20];
barrier q;

cx q[38], q[3];
cx q[45], q[17];
cx q[53], q[33];
cx q[41], q[9];
cx q[50], q[27];
cx q[39], q[4];
cx q[46], q[18];
cx q[54], q[34];
cx q[42], q[10];
cx q[51], q[28];
cx q[47], q[19];
cx q[43], q[11];
cx q[52], q[29];
cx q[48], q[20];
cx q[49], q[21];
barrier q;

cx q[38], q[5];
cx q[45], q[22];
cx q[53], q[35];
cx q[37], q[2];
cx q[41], q[13];
cx q[50], q[31];
cx q[39], q[6];
cx q[46], q[23];
cx q[42], q[14];
cx q[51], q[32];
cx q[40], q[7];
cx q[47], q[24];
cx q[43], q[15];
cx q[48], q[25];
cx q[44], q[16];
barrier q;

h q[38]; measure q[38] -> rec[108]; h q[38]; // decomposed MX
h q[45]; measure q[45] -> rec[109]; h q[45]; // decomposed MX
h q[53]; measure q[53] -> rec[110]; h q[53]; // decomposed MX
h q[37]; measure q[37] -> rec[111]; h q[37]; // decomposed MX
h q[41]; measure q[41] -> rec[112]; h q[41]; // decomposed MX
h q[50]; measure q[50] -> rec[113]; h q[50]; // decomposed MX
h q[39]; measure q[39] -> rec[114]; h q[39]; // decomposed MX
h q[46]; measure q[46] -> rec[115]; h q[46]; // decomposed MX
h q[54]; measure q[54] -> rec[116]; h q[54]; // decomposed MX
h q[42]; measure q[42] -> rec[117]; h q[42]; // decomposed MX
h q[51]; measure q[51] -> rec[118]; h q[51]; // decomposed MX
h q[40]; measure q[40] -> rec[119]; h q[40]; // decomposed MX
h q[47]; measure q[47] -> rec[120]; h q[47]; // decomposed MX
h q[43]; measure q[43] -> rec[121]; h q[43]; // decomposed MX
h q[52]; measure q[52] -> rec[122]; h q[52]; // decomposed MX
h q[48]; measure q[48] -> rec[123]; h q[48]; // decomposed MX
h q[44]; measure q[44] -> rec[124]; h q[44]; // decomposed MX
h q[49]; measure q[49] -> rec[125]; h q[49]; // decomposed MX
barrier q;

reset q[38];
reset q[45];
reset q[53];
reset q[37];
reset q[41];
reset q[50];
reset q[39];
reset q[46];
reset q[54];
reset q[42];
reset q[51];
reset q[40];
reset q[47];
reset q[43];
reset q[52];
reset q[48];
reset q[44];
reset q[49];
barrier q;

cx q[0], q[37];
cx q[8], q[41];
cx q[26], q[50];
cx q[3], q[39];
cx q[17], q[46];
cx q[33], q[54];
cx q[9], q[42];
cx q[27], q[51];
cx q[4], q[40];
cx q[18], q[47];
cx q[10], q[43];
cx q[28], q[52];
cx q[19], q[48];
cx q[11], q[44];
cx q[20], q[49];
barrier q;

cx q[1], q[37];
cx q[12], q[41];
cx q[30], q[50];
cx q[5], q[39];
cx q[22], q[46];
cx q[35], q[54];
cx q[13], q[42];
cx q[31], q[51];
cx q[6], q[40];
cx q[23], q[47];
cx q[14], q[43];
cx q[32], q[52];
cx q[24], q[48];
cx q[15], q[44];
cx q[25], q[49];
barrier q;

cx q[1], q[38];
cx q[12], q[45];
cx q[30], q[53];
cx q[5], q[41];
cx q[22], q[50];
cx q[2], q[39];
cx q[13], q[46];
cx q[31], q[54];
cx q[6], q[42];
cx q[23], q[51];
cx q[14], q[47];
cx q[7], q[43];
cx q[24], q[52];
cx q[15], q[48];
cx q[16], q[49];
barrier q;

cx q[8], q[38];
cx q[26], q[45];
cx q[36], q[53];
cx q[3], q[37];
cx q[17], q[41];
cx q[33], q[50];
cx q[9], q[39];
cx q[27], q[46];
cx q[18], q[42];
cx q[34], q[51];
cx q[10], q[40];
cx q[28], q[47];
cx q[19], q[43];
cx q[29], q[48];
cx q[20], q[44];
barrier q;

cx q[3], q[38];
cx q[17], q[45];
cx q[33], q[53];
cx q[9], q[41];
cx q[27], q[50];
cx q[4], q[39];
cx q[18], q[46];
cx q[34], q[54];
cx q[10], q[42];
cx q[28], q[51];
cx q[19], q[47];
cx q[11], q[43];
cx q[29], q[52];
cx q[20], q[48];
cx q[21], q[49];
barrier q;

cx q[5], q[38];
cx q[22], q[45];
cx q[35], q[53];
cx q[2], q[37];
cx q[13], q[41];
cx q[31], q[50];
cx q[6], q[39];
cx q[23], q[46];
cx q[14], q[42];
cx q[32], q[51];
cx q[7], q[40];
cx q[24], q[47];
cx q[15], q[43];
cx q[25], q[48];
cx q[16], q[44];
barrier q;

measure q[38] -> rec[126];
measure q[45] -> rec[127];
measure q[53] -> rec[128];
measure q[37] -> rec[129];
measure q[41] -> rec[130];
measure q[50] -> rec[131];
measure q[39] -> rec[132];
measure q[46] -> rec[133];
measure q[54] -> rec[134];
measure q[42] -> rec[135];
measure q[51] -> rec[136];
measure q[40] -> rec[137];
measure q[47] -> rec[138];
measure q[43] -> rec[139];
measure q[52] -> rec[140];
measure q[48] -> rec[141];
measure q[44] -> rec[142];
measure q[49] -> rec[143];
barrier q;

reset q[38]; h q[38]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[53]; h q[53]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[41]; h q[41]; // decomposed RX
reset q[50]; h q[50]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[46]; h q[46]; // decomposed RX
reset q[54]; h q[54]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
reset q[47]; h q[47]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[48]; h q[48]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[49]; h q[49]; // decomposed RX
barrier q;

cx q[37], q[0];
cx q[41], q[8];
cx q[50], q[26];
cx q[39], q[3];
cx q[46], q[17];
cx q[54], q[33];
cx q[42], q[9];
cx q[51], q[27];
cx q[40], q[4];
cx q[47], q[18];
cx q[43], q[10];
cx q[52], q[28];
cx q[48], q[19];
cx q[44], q[11];
cx q[49], q[20];
barrier q;

cx q[37], q[1];
cx q[41], q[12];
cx q[50], q[30];
cx q[39], q[5];
cx q[46], q[22];
cx q[54], q[35];
cx q[42], q[13];
cx q[51], q[31];
cx q[40], q[6];
cx q[47], q[23];
cx q[43], q[14];
cx q[52], q[32];
cx q[48], q[24];
cx q[44], q[15];
cx q[49], q[25];
barrier q;

cx q[38], q[1];
cx q[45], q[12];
cx q[53], q[30];
cx q[41], q[5];
cx q[50], q[22];
cx q[39], q[2];
cx q[46], q[13];
cx q[54], q[31];
cx q[42], q[6];
cx q[51], q[23];
cx q[47], q[14];
cx q[43], q[7];
cx q[52], q[24];
cx q[48], q[15];
cx q[49], q[16];
barrier q;

cx q[38], q[8];
cx q[45], q[26];
cx q[53], q[36];
cx q[37], q[3];
cx q[41], q[17];
cx q[50], q[33];
cx q[39], q[9];
cx q[46], q[27];
cx q[42], q[18];
cx q[51], q[34];
cx q[40], q[10];
cx q[47], q[28];
cx q[43], q[19];
cx q[48], q[29];
cx q[44], q[20];
barrier q;

cx q[38], q[3];
cx q[45], q[17];
cx q[53], q[33];
cx q[41], q[9];
cx q[50], q[27];
cx q[39], q[4];
cx q[46], q[18];
cx q[54], q[34];
cx q[42], q[10];
cx q[51], q[28];
cx q[47], q[19];
cx q[43], q[11];
cx q[52], q[29];
cx q[48], q[20];
cx q[49], q[21];
barrier q;

cx q[38], q[5];
cx q[45], q[22];
cx q[53], q[35];
cx q[37], q[2];
cx q[41], q[13];
cx q[50], q[31];
cx q[39], q[6];
cx q[46], q[23];
cx q[42], q[14];
cx q[51], q[32];
cx q[40], q[7];
cx q[47], q[24];
cx q[43], q[15];
cx q[48], q[25];
cx q[44], q[16];
barrier q;

h q[38]; measure q[38] -> rec[144]; h q[38]; // decomposed MX
h q[45]; measure q[45] -> rec[145]; h q[45]; // decomposed MX
h q[53]; measure q[53] -> rec[146]; h q[53]; // decomposed MX
h q[37]; measure q[37] -> rec[147]; h q[37]; // decomposed MX
h q[41]; measure q[41] -> rec[148]; h q[41]; // decomposed MX
h q[50]; measure q[50] -> rec[149]; h q[50]; // decomposed MX
h q[39]; measure q[39] -> rec[150]; h q[39]; // decomposed MX
h q[46]; measure q[46] -> rec[151]; h q[46]; // decomposed MX
h q[54]; measure q[54] -> rec[152]; h q[54]; // decomposed MX
h q[42]; measure q[42] -> rec[153]; h q[42]; // decomposed MX
h q[51]; measure q[51] -> rec[154]; h q[51]; // decomposed MX
h q[40]; measure q[40] -> rec[155]; h q[40]; // decomposed MX
h q[47]; measure q[47] -> rec[156]; h q[47]; // decomposed MX
h q[43]; measure q[43] -> rec[157]; h q[43]; // decomposed MX
h q[52]; measure q[52] -> rec[158]; h q[52]; // decomposed MX
h q[48]; measure q[48] -> rec[159]; h q[48]; // decomposed MX
h q[44]; measure q[44] -> rec[160]; h q[44]; // decomposed MX
h q[49]; measure q[49] -> rec[161]; h q[49]; // decomposed MX
barrier q;

reset q[38];
reset q[45];
reset q[53];
reset q[37];
reset q[41];
reset q[50];
reset q[39];
reset q[46];
reset q[54];
reset q[42];
reset q[51];
reset q[40];
reset q[47];
reset q[43];
reset q[52];
reset q[48];
reset q[44];
reset q[49];
barrier q;

cx q[0], q[37];
cx q[8], q[41];
cx q[26], q[50];
cx q[3], q[39];
cx q[17], q[46];
cx q[33], q[54];
cx q[9], q[42];
cx q[27], q[51];
cx q[4], q[40];
cx q[18], q[47];
cx q[10], q[43];
cx q[28], q[52];
cx q[19], q[48];
cx q[11], q[44];
cx q[20], q[49];
barrier q;

cx q[1], q[37];
cx q[12], q[41];
cx q[30], q[50];
cx q[5], q[39];
cx q[22], q[46];
cx q[35], q[54];
cx q[13], q[42];
cx q[31], q[51];
cx q[6], q[40];
cx q[23], q[47];
cx q[14], q[43];
cx q[32], q[52];
cx q[24], q[48];
cx q[15], q[44];
cx q[25], q[49];
barrier q;

cx q[1], q[38];
cx q[12], q[45];
cx q[30], q[53];
cx q[5], q[41];
cx q[22], q[50];
cx q[2], q[39];
cx q[13], q[46];
cx q[31], q[54];
cx q[6], q[42];
cx q[23], q[51];
cx q[14], q[47];
cx q[7], q[43];
cx q[24], q[52];
cx q[15], q[48];
cx q[16], q[49];
barrier q;

cx q[8], q[38];
cx q[26], q[45];
cx q[36], q[53];
cx q[3], q[37];
cx q[17], q[41];
cx q[33], q[50];
cx q[9], q[39];
cx q[27], q[46];
cx q[18], q[42];
cx q[34], q[51];
cx q[10], q[40];
cx q[28], q[47];
cx q[19], q[43];
cx q[29], q[48];
cx q[20], q[44];
barrier q;

cx q[3], q[38];
cx q[17], q[45];
cx q[33], q[53];
cx q[9], q[41];
cx q[27], q[50];
cx q[4], q[39];
cx q[18], q[46];
cx q[34], q[54];
cx q[10], q[42];
cx q[28], q[51];
cx q[19], q[47];
cx q[11], q[43];
cx q[29], q[52];
cx q[20], q[48];
cx q[21], q[49];
barrier q;

cx q[5], q[38];
cx q[22], q[45];
cx q[35], q[53];
cx q[2], q[37];
cx q[13], q[41];
cx q[31], q[50];
cx q[6], q[39];
cx q[23], q[46];
cx q[14], q[42];
cx q[32], q[51];
cx q[7], q[40];
cx q[24], q[47];
cx q[15], q[43];
cx q[25], q[48];
cx q[16], q[44];
barrier q;

measure q[38] -> rec[162];
measure q[45] -> rec[163];
measure q[53] -> rec[164];
measure q[37] -> rec[165];
measure q[41] -> rec[166];
measure q[50] -> rec[167];
measure q[39] -> rec[168];
measure q[46] -> rec[169];
measure q[54] -> rec[170];
measure q[42] -> rec[171];
measure q[51] -> rec[172];
measure q[40] -> rec[173];
measure q[47] -> rec[174];
measure q[43] -> rec[175];
measure q[52] -> rec[176];
measure q[48] -> rec[177];
measure q[44] -> rec[178];
measure q[49] -> rec[179];
barrier q;

measure q[0] -> rec[180];
measure q[1] -> rec[181];
measure q[2] -> rec[182];
measure q[3] -> rec[183];
measure q[4] -> rec[184];
measure q[5] -> rec[185];
measure q[6] -> rec[186];
measure q[7] -> rec[187];
measure q[8] -> rec[188];
measure q[9] -> rec[189];
measure q[10] -> rec[190];
measure q[11] -> rec[191];
measure q[12] -> rec[192];
measure q[13] -> rec[193];
measure q[14] -> rec[194];
measure q[15] -> rec[195];
measure q[16] -> rec[196];
measure q[17] -> rec[197];
measure q[18] -> rec[198];
measure q[19] -> rec[199];
measure q[20] -> rec[200];
measure q[21] -> rec[201];
measure q[22] -> rec[202];
measure q[23] -> rec[203];
measure q[24] -> rec[204];
measure q[25] -> rec[205];
measure q[26] -> rec[206];
measure q[27] -> rec[207];
measure q[28] -> rec[208];
measure q[29] -> rec[209];
measure q[30] -> rec[210];
measure q[31] -> rec[211];
measure q[32] -> rec[212];
measure q[33] -> rec[213];
measure q[34] -> rec[214];
measure q[35] -> rec[215];
measure q[36] -> rec[216];
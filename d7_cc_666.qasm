OPENQASM 2.0;
include "qelib1.inc";

qreg q[55];
creg rec[217];

reset q[0];
reset q[2];
reset q[3];
reset q[4];
reset q[5];
reset q[9];
reset q[10];
reset q[11];
reset q[12];
reset q[13];
reset q[14];
reset q[15];
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
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
reset q[41];
reset q[42];
reset q[46];
reset q[47];
reset q[48];
reset q[49];
reset q[50];
reset q[53];
reset q[54];
reset q[32];
reset q[1];
reset q[34];
reset q[33];
reset q[6];
reset q[7];
reset q[8];
reset q[43];
reset q[44];
reset q[45];
reset q[16];
reset q[17];
reset q[18];
reset q[19];
reset q[51];
reset q[52];
reset q[30];
reset q[31];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[30]; h q[30]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[16]; h q[16]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[17]; h q[17]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[8]; h q[8]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[16], q[12];
cx q[43], q[39];
cx q[7], q[4];
cx q[31], q[25];
cx q[52], q[49];
cx q[17], q[13];
cx q[44], q[40];
cx q[8], q[5];
cx q[32], q[26];
cx q[18], q[14];
cx q[45], q[41];
cx q[33], q[27];
cx q[19], q[15];
cx q[34], q[28];
barrier q;

cx q[1], q[2];
cx q[16], q[20];
cx q[43], q[46];
cx q[7], q[9];
cx q[31], q[35];
cx q[52], q[53];
cx q[17], q[21];
cx q[44], q[47];
cx q[8], q[10];
cx q[32], q[36];
cx q[18], q[22];
cx q[45], q[48];
cx q[33], q[37];
cx q[19], q[23];
cx q[34], q[38];
barrier q;

cx q[6], q[2];
cx q[30], q[20];
cx q[51], q[46];
cx q[16], q[9];
cx q[43], q[35];
cx q[7], q[3];
cx q[31], q[21];
cx q[52], q[47];
cx q[17], q[10];
cx q[44], q[36];
cx q[32], q[22];
cx q[18], q[11];
cx q[45], q[37];
cx q[33], q[23];
cx q[34], q[24];
barrier q;

cx q[6], q[12];
cx q[30], q[39];
cx q[51], q[54];
cx q[1], q[4];
cx q[16], q[25];
cx q[43], q[49];
cx q[7], q[13];
cx q[31], q[40];
cx q[17], q[26];
cx q[44], q[50];
cx q[8], q[14];
cx q[32], q[41];
cx q[18], q[27];
cx q[33], q[42];
cx q[19], q[28];
barrier q;

cx q[6], q[4];
cx q[30], q[25];
cx q[51], q[49];
cx q[16], q[13];
cx q[43], q[40];
cx q[7], q[5];
cx q[31], q[26];
cx q[52], q[50];
cx q[17], q[14];
cx q[44], q[41];
cx q[32], q[27];
cx q[18], q[15];
cx q[45], q[42];
cx q[33], q[28];
cx q[34], q[29];
barrier q;

cx q[6], q[9];
cx q[30], q[35];
cx q[51], q[53];
cx q[1], q[3];
cx q[16], q[21];
cx q[43], q[47];
cx q[7], q[10];
cx q[31], q[36];
cx q[17], q[22];
cx q[44], q[48];
cx q[8], q[11];
cx q[32], q[37];
cx q[18], q[23];
cx q[33], q[38];
cx q[19], q[24];
barrier q;

h q[6]; measure q[6] -> rec[0]; h q[6]; // decomposed MX
h q[30]; measure q[30] -> rec[1]; h q[30]; // decomposed MX
h q[51]; measure q[51] -> rec[2]; h q[51]; // decomposed MX
h q[1]; measure q[1] -> rec[3]; h q[1]; // decomposed MX
h q[16]; measure q[16] -> rec[4]; h q[16]; // decomposed MX
h q[43]; measure q[43] -> rec[5]; h q[43]; // decomposed MX
h q[7]; measure q[7] -> rec[6]; h q[7]; // decomposed MX
h q[31]; measure q[31] -> rec[7]; h q[31]; // decomposed MX
h q[52]; measure q[52] -> rec[8]; h q[52]; // decomposed MX
h q[17]; measure q[17] -> rec[9]; h q[17]; // decomposed MX
h q[44]; measure q[44] -> rec[10]; h q[44]; // decomposed MX
h q[8]; measure q[8] -> rec[11]; h q[8]; // decomposed MX
h q[32]; measure q[32] -> rec[12]; h q[32]; // decomposed MX
h q[18]; measure q[18] -> rec[13]; h q[18]; // decomposed MX
h q[45]; measure q[45] -> rec[14]; h q[45]; // decomposed MX
h q[33]; measure q[33] -> rec[15]; h q[33]; // decomposed MX
h q[19]; measure q[19] -> rec[16]; h q[19]; // decomposed MX
h q[34]; measure q[34] -> rec[17]; h q[34]; // decomposed MX
reset q[6];
reset q[30];
reset q[51];
reset q[1];
reset q[16];
reset q[43];
reset q[7];
reset q[31];
reset q[52];
reset q[17];
reset q[44];
reset q[8];
reset q[32];
reset q[18];
reset q[45];
reset q[33];
reset q[19];
reset q[34];
barrier q;

cx q[0], q[1];
cx q[12], q[16];
cx q[39], q[43];
cx q[4], q[7];
cx q[25], q[31];
cx q[49], q[52];
cx q[13], q[17];
cx q[40], q[44];
cx q[5], q[8];
cx q[26], q[32];
cx q[14], q[18];
cx q[41], q[45];
cx q[27], q[33];
cx q[15], q[19];
cx q[28], q[34];
barrier q;

cx q[2], q[1];
cx q[20], q[16];
cx q[46], q[43];
cx q[9], q[7];
cx q[35], q[31];
cx q[53], q[52];
cx q[21], q[17];
cx q[47], q[44];
cx q[10], q[8];
cx q[36], q[32];
cx q[22], q[18];
cx q[48], q[45];
cx q[37], q[33];
cx q[23], q[19];
cx q[38], q[34];
barrier q;

cx q[2], q[6];
cx q[20], q[30];
cx q[46], q[51];
cx q[9], q[16];
cx q[35], q[43];
cx q[3], q[7];
cx q[21], q[31];
cx q[47], q[52];
cx q[10], q[17];
cx q[36], q[44];
cx q[22], q[32];
cx q[11], q[18];
cx q[37], q[45];
cx q[23], q[33];
cx q[24], q[34];
barrier q;

cx q[12], q[6];
cx q[39], q[30];
cx q[54], q[51];
cx q[4], q[1];
cx q[25], q[16];
cx q[49], q[43];
cx q[13], q[7];
cx q[40], q[31];
cx q[26], q[17];
cx q[50], q[44];
cx q[14], q[8];
cx q[41], q[32];
cx q[27], q[18];
cx q[42], q[33];
cx q[28], q[19];
barrier q;

cx q[4], q[6];
cx q[25], q[30];
cx q[49], q[51];
cx q[13], q[16];
cx q[40], q[43];
cx q[5], q[7];
cx q[26], q[31];
cx q[50], q[52];
cx q[14], q[17];
cx q[41], q[44];
cx q[27], q[32];
cx q[15], q[18];
cx q[42], q[45];
cx q[28], q[33];
cx q[29], q[34];
barrier q;

cx q[9], q[6];
cx q[35], q[30];
cx q[53], q[51];
cx q[3], q[1];
cx q[21], q[16];
cx q[47], q[43];
cx q[10], q[7];
cx q[36], q[31];
cx q[22], q[17];
cx q[48], q[44];
cx q[11], q[8];
cx q[37], q[32];
cx q[23], q[18];
cx q[38], q[33];
cx q[24], q[19];
barrier q;

measure q[6] -> rec[18];
measure q[30] -> rec[19];
measure q[51] -> rec[20];
measure q[1] -> rec[21];
measure q[16] -> rec[22];
measure q[43] -> rec[23];
measure q[7] -> rec[24];
measure q[31] -> rec[25];
measure q[52] -> rec[26];
measure q[17] -> rec[27];
measure q[44] -> rec[28];
measure q[8] -> rec[29];
measure q[32] -> rec[30];
measure q[18] -> rec[31];
measure q[45] -> rec[32];
measure q[33] -> rec[33];
measure q[19] -> rec[34];
measure q[34] -> rec[35];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[30]; h q[30]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[16]; h q[16]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[17]; h q[17]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[8]; h q[8]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[16], q[12];
cx q[43], q[39];
cx q[7], q[4];
cx q[31], q[25];
cx q[52], q[49];
cx q[17], q[13];
cx q[44], q[40];
cx q[8], q[5];
cx q[32], q[26];
cx q[18], q[14];
cx q[45], q[41];
cx q[33], q[27];
cx q[19], q[15];
cx q[34], q[28];
barrier q;

cx q[1], q[2];
cx q[16], q[20];
cx q[43], q[46];
cx q[7], q[9];
cx q[31], q[35];
cx q[52], q[53];
cx q[17], q[21];
cx q[44], q[47];
cx q[8], q[10];
cx q[32], q[36];
cx q[18], q[22];
cx q[45], q[48];
cx q[33], q[37];
cx q[19], q[23];
cx q[34], q[38];
barrier q;

cx q[6], q[2];
cx q[30], q[20];
cx q[51], q[46];
cx q[16], q[9];
cx q[43], q[35];
cx q[7], q[3];
cx q[31], q[21];
cx q[52], q[47];
cx q[17], q[10];
cx q[44], q[36];
cx q[32], q[22];
cx q[18], q[11];
cx q[45], q[37];
cx q[33], q[23];
cx q[34], q[24];
barrier q;

cx q[6], q[12];
cx q[30], q[39];
cx q[51], q[54];
cx q[1], q[4];
cx q[16], q[25];
cx q[43], q[49];
cx q[7], q[13];
cx q[31], q[40];
cx q[17], q[26];
cx q[44], q[50];
cx q[8], q[14];
cx q[32], q[41];
cx q[18], q[27];
cx q[33], q[42];
cx q[19], q[28];
barrier q;

cx q[6], q[4];
cx q[30], q[25];
cx q[51], q[49];
cx q[16], q[13];
cx q[43], q[40];
cx q[7], q[5];
cx q[31], q[26];
cx q[52], q[50];
cx q[17], q[14];
cx q[44], q[41];
cx q[32], q[27];
cx q[18], q[15];
cx q[45], q[42];
cx q[33], q[28];
cx q[34], q[29];
barrier q;

cx q[6], q[9];
cx q[30], q[35];
cx q[51], q[53];
cx q[1], q[3];
cx q[16], q[21];
cx q[43], q[47];
cx q[7], q[10];
cx q[31], q[36];
cx q[17], q[22];
cx q[44], q[48];
cx q[8], q[11];
cx q[32], q[37];
cx q[18], q[23];
cx q[33], q[38];
cx q[19], q[24];
barrier q;

h q[6]; measure q[6] -> rec[36]; h q[6]; // decomposed MX
h q[30]; measure q[30] -> rec[37]; h q[30]; // decomposed MX
h q[51]; measure q[51] -> rec[38]; h q[51]; // decomposed MX
h q[1]; measure q[1] -> rec[39]; h q[1]; // decomposed MX
h q[16]; measure q[16] -> rec[40]; h q[16]; // decomposed MX
h q[43]; measure q[43] -> rec[41]; h q[43]; // decomposed MX
h q[7]; measure q[7] -> rec[42]; h q[7]; // decomposed MX
h q[31]; measure q[31] -> rec[43]; h q[31]; // decomposed MX
h q[52]; measure q[52] -> rec[44]; h q[52]; // decomposed MX
h q[17]; measure q[17] -> rec[45]; h q[17]; // decomposed MX
h q[44]; measure q[44] -> rec[46]; h q[44]; // decomposed MX
h q[8]; measure q[8] -> rec[47]; h q[8]; // decomposed MX
h q[32]; measure q[32] -> rec[48]; h q[32]; // decomposed MX
h q[18]; measure q[18] -> rec[49]; h q[18]; // decomposed MX
h q[45]; measure q[45] -> rec[50]; h q[45]; // decomposed MX
h q[33]; measure q[33] -> rec[51]; h q[33]; // decomposed MX
h q[19]; measure q[19] -> rec[52]; h q[19]; // decomposed MX
h q[34]; measure q[34] -> rec[53]; h q[34]; // decomposed MX
barrier q;

reset q[6];
reset q[30];
reset q[51];
reset q[1];
reset q[16];
reset q[43];
reset q[7];
reset q[31];
reset q[52];
reset q[17];
reset q[44];
reset q[8];
reset q[32];
reset q[18];
reset q[45];
reset q[33];
reset q[19];
reset q[34];
barrier q;

cx q[0], q[1];
cx q[12], q[16];
cx q[39], q[43];
cx q[4], q[7];
cx q[25], q[31];
cx q[49], q[52];
cx q[13], q[17];
cx q[40], q[44];
cx q[5], q[8];
cx q[26], q[32];
cx q[14], q[18];
cx q[41], q[45];
cx q[27], q[33];
cx q[15], q[19];
cx q[28], q[34];
barrier q;

cx q[2], q[1];
cx q[20], q[16];
cx q[46], q[43];
cx q[9], q[7];
cx q[35], q[31];
cx q[53], q[52];
cx q[21], q[17];
cx q[47], q[44];
cx q[10], q[8];
cx q[36], q[32];
cx q[22], q[18];
cx q[48], q[45];
cx q[37], q[33];
cx q[23], q[19];
cx q[38], q[34];
barrier q;

cx q[2], q[6];
cx q[20], q[30];
cx q[46], q[51];
cx q[9], q[16];
cx q[35], q[43];
cx q[3], q[7];
cx q[21], q[31];
cx q[47], q[52];
cx q[10], q[17];
cx q[36], q[44];
cx q[22], q[32];
cx q[11], q[18];
cx q[37], q[45];
cx q[23], q[33];
cx q[24], q[34];
barrier q;

cx q[12], q[6];
cx q[39], q[30];
cx q[54], q[51];
cx q[4], q[1];
cx q[25], q[16];
cx q[49], q[43];
cx q[13], q[7];
cx q[40], q[31];
cx q[26], q[17];
cx q[50], q[44];
cx q[14], q[8];
cx q[41], q[32];
cx q[27], q[18];
cx q[42], q[33];
cx q[28], q[19];
barrier q;

cx q[4], q[6];
cx q[25], q[30];
cx q[49], q[51];
cx q[13], q[16];
cx q[40], q[43];
cx q[5], q[7];
cx q[26], q[31];
cx q[50], q[52];
cx q[14], q[17];
cx q[41], q[44];
cx q[27], q[32];
cx q[15], q[18];
cx q[42], q[45];
cx q[28], q[33];
cx q[29], q[34];
barrier q;

cx q[9], q[6];
cx q[35], q[30];
cx q[53], q[51];
cx q[3], q[1];
cx q[21], q[16];
cx q[47], q[43];
cx q[10], q[7];
cx q[36], q[31];
cx q[22], q[17];
cx q[48], q[44];
cx q[11], q[8];
cx q[37], q[32];
cx q[23], q[18];
cx q[38], q[33];
cx q[24], q[19];
barrier q;

measure q[6] -> rec[54];
measure q[30] -> rec[55];
measure q[51] -> rec[56];
measure q[1] -> rec[57];
measure q[16] -> rec[58];
measure q[43] -> rec[59];
measure q[7] -> rec[60];
measure q[31] -> rec[61];
measure q[52] -> rec[62];
measure q[17] -> rec[63];
measure q[44] -> rec[64];
measure q[8] -> rec[65];
measure q[32] -> rec[66];
measure q[18] -> rec[67];
measure q[45] -> rec[68];
measure q[33] -> rec[69];
measure q[19] -> rec[70];
measure q[34] -> rec[71];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[30]; h q[30]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[16]; h q[16]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[17]; h q[17]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[8]; h q[8]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[16], q[12];
cx q[43], q[39];
cx q[7], q[4];
cx q[31], q[25];
cx q[52], q[49];
cx q[17], q[13];
cx q[44], q[40];
cx q[8], q[5];
cx q[32], q[26];
cx q[18], q[14];
cx q[45], q[41];
cx q[33], q[27];
cx q[19], q[15];
cx q[34], q[28];
barrier q;

cx q[1], q[2];
cx q[16], q[20];
cx q[43], q[46];
cx q[7], q[9];
cx q[31], q[35];
cx q[52], q[53];
cx q[17], q[21];
cx q[44], q[47];
cx q[8], q[10];
cx q[32], q[36];
cx q[18], q[22];
cx q[45], q[48];
cx q[33], q[37];
cx q[19], q[23];
cx q[34], q[38];
barrier q;

cx q[6], q[2];
cx q[30], q[20];
cx q[51], q[46];
cx q[16], q[9];
cx q[43], q[35];
cx q[7], q[3];
cx q[31], q[21];
cx q[52], q[47];
cx q[17], q[10];
cx q[44], q[36];
cx q[32], q[22];
cx q[18], q[11];
cx q[45], q[37];
cx q[33], q[23];
cx q[34], q[24];
barrier q;

cx q[6], q[12];
cx q[30], q[39];
cx q[51], q[54];
cx q[1], q[4];
cx q[16], q[25];
cx q[43], q[49];
cx q[7], q[13];
cx q[31], q[40];
cx q[17], q[26];
cx q[44], q[50];
cx q[8], q[14];
cx q[32], q[41];
cx q[18], q[27];
cx q[33], q[42];
cx q[19], q[28];
barrier q;

cx q[6], q[4];
cx q[30], q[25];
cx q[51], q[49];
cx q[16], q[13];
cx q[43], q[40];
cx q[7], q[5];
cx q[31], q[26];
cx q[52], q[50];
cx q[17], q[14];
cx q[44], q[41];
cx q[32], q[27];
cx q[18], q[15];
cx q[45], q[42];
cx q[33], q[28];
cx q[34], q[29];
barrier q;

cx q[6], q[9];
cx q[30], q[35];
cx q[51], q[53];
cx q[1], q[3];
cx q[16], q[21];
cx q[43], q[47];
cx q[7], q[10];
cx q[31], q[36];
cx q[17], q[22];
cx q[44], q[48];
cx q[8], q[11];
cx q[32], q[37];
cx q[18], q[23];
cx q[33], q[38];
cx q[19], q[24];
barrier q;

h q[6]; measure q[6] -> rec[72]; h q[6]; // decomposed MX
h q[30]; measure q[30] -> rec[73]; h q[30]; // decomposed MX
h q[51]; measure q[51] -> rec[74]; h q[51]; // decomposed MX
h q[1]; measure q[1] -> rec[75]; h q[1]; // decomposed MX
h q[16]; measure q[16] -> rec[76]; h q[16]; // decomposed MX
h q[43]; measure q[43] -> rec[77]; h q[43]; // decomposed MX
h q[7]; measure q[7] -> rec[78]; h q[7]; // decomposed MX
h q[31]; measure q[31] -> rec[79]; h q[31]; // decomposed MX
h q[52]; measure q[52] -> rec[80]; h q[52]; // decomposed MX
h q[17]; measure q[17] -> rec[81]; h q[17]; // decomposed MX
h q[44]; measure q[44] -> rec[82]; h q[44]; // decomposed MX
h q[8]; measure q[8] -> rec[83]; h q[8]; // decomposed MX
h q[32]; measure q[32] -> rec[84]; h q[32]; // decomposed MX
h q[18]; measure q[18] -> rec[85]; h q[18]; // decomposed MX
h q[45]; measure q[45] -> rec[86]; h q[45]; // decomposed MX
h q[33]; measure q[33] -> rec[87]; h q[33]; // decomposed MX
h q[19]; measure q[19] -> rec[88]; h q[19]; // decomposed MX
h q[34]; measure q[34] -> rec[89]; h q[34]; // decomposed MX
barrier q;

reset q[6];
reset q[30];
reset q[51];
reset q[1];
reset q[16];
reset q[43];
reset q[7];
reset q[31];
reset q[52];
reset q[17];
reset q[44];
reset q[8];
reset q[32];
reset q[18];
reset q[45];
reset q[33];
reset q[19];
reset q[34];
barrier q;

cx q[0], q[1];
cx q[12], q[16];
cx q[39], q[43];
cx q[4], q[7];
cx q[25], q[31];
cx q[49], q[52];
cx q[13], q[17];
cx q[40], q[44];
cx q[5], q[8];
cx q[26], q[32];
cx q[14], q[18];
cx q[41], q[45];
cx q[27], q[33];
cx q[15], q[19];
cx q[28], q[34];
barrier q;

cx q[2], q[1];
cx q[20], q[16];
cx q[46], q[43];
cx q[9], q[7];
cx q[35], q[31];
cx q[53], q[52];
cx q[21], q[17];
cx q[47], q[44];
cx q[10], q[8];
cx q[36], q[32];
cx q[22], q[18];
cx q[48], q[45];
cx q[37], q[33];
cx q[23], q[19];
cx q[38], q[34];
barrier q;

cx q[2], q[6];
cx q[20], q[30];
cx q[46], q[51];
cx q[9], q[16];
cx q[35], q[43];
cx q[3], q[7];
cx q[21], q[31];
cx q[47], q[52];
cx q[10], q[17];
cx q[36], q[44];
cx q[22], q[32];
cx q[11], q[18];
cx q[37], q[45];
cx q[23], q[33];
cx q[24], q[34];
barrier q;

cx q[12], q[6];
cx q[39], q[30];
cx q[54], q[51];
cx q[4], q[1];
cx q[25], q[16];
cx q[49], q[43];
cx q[13], q[7];
cx q[40], q[31];
cx q[26], q[17];
cx q[50], q[44];
cx q[14], q[8];
cx q[41], q[32];
cx q[27], q[18];
cx q[42], q[33];
cx q[28], q[19];
barrier q;

cx q[4], q[6];
cx q[25], q[30];
cx q[49], q[51];
cx q[13], q[16];
cx q[40], q[43];
cx q[5], q[7];
cx q[26], q[31];
cx q[50], q[52];
cx q[14], q[17];
cx q[41], q[44];
cx q[27], q[32];
cx q[15], q[18];
cx q[42], q[45];
cx q[28], q[33];
cx q[29], q[34];
barrier q;

cx q[9], q[6];
cx q[35], q[30];
cx q[53], q[51];
cx q[3], q[1];
cx q[21], q[16];
cx q[47], q[43];
cx q[10], q[7];
cx q[36], q[31];
cx q[22], q[17];
cx q[48], q[44];
cx q[11], q[8];
cx q[37], q[32];
cx q[23], q[18];
cx q[38], q[33];
cx q[24], q[19];
barrier q;

measure q[6] -> rec[90];
measure q[30] -> rec[91];
measure q[51] -> rec[92];
measure q[1] -> rec[93];
measure q[16] -> rec[94];
measure q[43] -> rec[95];
measure q[7] -> rec[96];
measure q[31] -> rec[97];
measure q[52] -> rec[98];
measure q[17] -> rec[99];
measure q[44] -> rec[100];
measure q[8] -> rec[101];
measure q[32] -> rec[102];
measure q[18] -> rec[103];
measure q[45] -> rec[104];
measure q[33] -> rec[105];
measure q[19] -> rec[106];
measure q[34] -> rec[107];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[30]; h q[30]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[16]; h q[16]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[17]; h q[17]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[8]; h q[8]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[16], q[12];
cx q[43], q[39];
cx q[7], q[4];
cx q[31], q[25];
cx q[52], q[49];
cx q[17], q[13];
cx q[44], q[40];
cx q[8], q[5];
cx q[32], q[26];
cx q[18], q[14];
cx q[45], q[41];
cx q[33], q[27];
cx q[19], q[15];
cx q[34], q[28];
barrier q;

cx q[1], q[2];
cx q[16], q[20];
cx q[43], q[46];
cx q[7], q[9];
cx q[31], q[35];
cx q[52], q[53];
cx q[17], q[21];
cx q[44], q[47];
cx q[8], q[10];
cx q[32], q[36];
cx q[18], q[22];
cx q[45], q[48];
cx q[33], q[37];
cx q[19], q[23];
cx q[34], q[38];
barrier q;

cx q[6], q[2];
cx q[30], q[20];
cx q[51], q[46];
cx q[16], q[9];
cx q[43], q[35];
cx q[7], q[3];
cx q[31], q[21];
cx q[52], q[47];
cx q[17], q[10];
cx q[44], q[36];
cx q[32], q[22];
cx q[18], q[11];
cx q[45], q[37];
cx q[33], q[23];
cx q[34], q[24];
barrier q;

cx q[6], q[12];
cx q[30], q[39];
cx q[51], q[54];
cx q[1], q[4];
cx q[16], q[25];
cx q[43], q[49];
cx q[7], q[13];
cx q[31], q[40];
cx q[17], q[26];
cx q[44], q[50];
cx q[8], q[14];
cx q[32], q[41];
cx q[18], q[27];
cx q[33], q[42];
cx q[19], q[28];
barrier q;

cx q[6], q[4];
cx q[30], q[25];
cx q[51], q[49];
cx q[16], q[13];
cx q[43], q[40];
cx q[7], q[5];
cx q[31], q[26];
cx q[52], q[50];
cx q[17], q[14];
cx q[44], q[41];
cx q[32], q[27];
cx q[18], q[15];
cx q[45], q[42];
cx q[33], q[28];
cx q[34], q[29];
barrier q;

cx q[6], q[9];
cx q[30], q[35];
cx q[51], q[53];
cx q[1], q[3];
cx q[16], q[21];
cx q[43], q[47];
cx q[7], q[10];
cx q[31], q[36];
cx q[17], q[22];
cx q[44], q[48];
cx q[8], q[11];
cx q[32], q[37];
cx q[18], q[23];
cx q[33], q[38];
cx q[19], q[24];
barrier q;

h q[6]; measure q[6] -> rec[108]; h q[6]; // decomposed MX
h q[30]; measure q[30] -> rec[109]; h q[30]; // decomposed MX
h q[51]; measure q[51] -> rec[110]; h q[51]; // decomposed MX
h q[1]; measure q[1] -> rec[111]; h q[1]; // decomposed MX
h q[16]; measure q[16] -> rec[112]; h q[16]; // decomposed MX
h q[43]; measure q[43] -> rec[113]; h q[43]; // decomposed MX
h q[7]; measure q[7] -> rec[114]; h q[7]; // decomposed MX
h q[31]; measure q[31] -> rec[115]; h q[31]; // decomposed MX
h q[52]; measure q[52] -> rec[116]; h q[52]; // decomposed MX
h q[17]; measure q[17] -> rec[117]; h q[17]; // decomposed MX
h q[44]; measure q[44] -> rec[118]; h q[44]; // decomposed MX
h q[8]; measure q[8] -> rec[119]; h q[8]; // decomposed MX
h q[32]; measure q[32] -> rec[120]; h q[32]; // decomposed MX
h q[18]; measure q[18] -> rec[121]; h q[18]; // decomposed MX
h q[45]; measure q[45] -> rec[122]; h q[45]; // decomposed MX
h q[33]; measure q[33] -> rec[123]; h q[33]; // decomposed MX
h q[19]; measure q[19] -> rec[124]; h q[19]; // decomposed MX
h q[34]; measure q[34] -> rec[125]; h q[34]; // decomposed MX
barrier q;

reset q[6];
reset q[30];
reset q[51];
reset q[1];
reset q[16];
reset q[43];
reset q[7];
reset q[31];
reset q[52];
reset q[17];
reset q[44];
reset q[8];
reset q[32];
reset q[18];
reset q[45];
reset q[33];
reset q[19];
reset q[34];
barrier q;

cx q[0], q[1];
cx q[12], q[16];
cx q[39], q[43];
cx q[4], q[7];
cx q[25], q[31];
cx q[49], q[52];
cx q[13], q[17];
cx q[40], q[44];
cx q[5], q[8];
cx q[26], q[32];
cx q[14], q[18];
cx q[41], q[45];
cx q[27], q[33];
cx q[15], q[19];
cx q[28], q[34];
barrier q;

cx q[2], q[1];
cx q[20], q[16];
cx q[46], q[43];
cx q[9], q[7];
cx q[35], q[31];
cx q[53], q[52];
cx q[21], q[17];
cx q[47], q[44];
cx q[10], q[8];
cx q[36], q[32];
cx q[22], q[18];
cx q[48], q[45];
cx q[37], q[33];
cx q[23], q[19];
cx q[38], q[34];
barrier q;

cx q[2], q[6];
cx q[20], q[30];
cx q[46], q[51];
cx q[9], q[16];
cx q[35], q[43];
cx q[3], q[7];
cx q[21], q[31];
cx q[47], q[52];
cx q[10], q[17];
cx q[36], q[44];
cx q[22], q[32];
cx q[11], q[18];
cx q[37], q[45];
cx q[23], q[33];
cx q[24], q[34];
barrier q;

cx q[12], q[6];
cx q[39], q[30];
cx q[54], q[51];
cx q[4], q[1];
cx q[25], q[16];
cx q[49], q[43];
cx q[13], q[7];
cx q[40], q[31];
cx q[26], q[17];
cx q[50], q[44];
cx q[14], q[8];
cx q[41], q[32];
cx q[27], q[18];
cx q[42], q[33];
cx q[28], q[19];
barrier q;

cx q[4], q[6];
cx q[25], q[30];
cx q[49], q[51];
cx q[13], q[16];
cx q[40], q[43];
cx q[5], q[7];
cx q[26], q[31];
cx q[50], q[52];
cx q[14], q[17];
cx q[41], q[44];
cx q[27], q[32];
cx q[15], q[18];
cx q[42], q[45];
cx q[28], q[33];
cx q[29], q[34];
barrier q;

cx q[9], q[6];
cx q[35], q[30];
cx q[53], q[51];
cx q[3], q[1];
cx q[21], q[16];
cx q[47], q[43];
cx q[10], q[7];
cx q[36], q[31];
cx q[22], q[17];
cx q[48], q[44];
cx q[11], q[8];
cx q[37], q[32];
cx q[23], q[18];
cx q[38], q[33];
cx q[24], q[19];
barrier q;

measure q[6] -> rec[126];
measure q[30] -> rec[127];
measure q[51] -> rec[128];
measure q[1] -> rec[129];
measure q[16] -> rec[130];
measure q[43] -> rec[131];
measure q[7] -> rec[132];
measure q[31] -> rec[133];
measure q[52] -> rec[134];
measure q[17] -> rec[135];
measure q[44] -> rec[136];
measure q[8] -> rec[137];
measure q[32] -> rec[138];
measure q[18] -> rec[139];
measure q[45] -> rec[140];
measure q[33] -> rec[141];
measure q[19] -> rec[142];
measure q[34] -> rec[143];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[30]; h q[30]; // decomposed RX
reset q[51]; h q[51]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[16]; h q[16]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[52]; h q[52]; // decomposed RX
reset q[17]; h q[17]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[8]; h q[8]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[16], q[12];
cx q[43], q[39];
cx q[7], q[4];
cx q[31], q[25];
cx q[52], q[49];
cx q[17], q[13];
cx q[44], q[40];
cx q[8], q[5];
cx q[32], q[26];
cx q[18], q[14];
cx q[45], q[41];
cx q[33], q[27];
cx q[19], q[15];
cx q[34], q[28];
barrier q;

cx q[1], q[2];
cx q[16], q[20];
cx q[43], q[46];
cx q[7], q[9];
cx q[31], q[35];
cx q[52], q[53];
cx q[17], q[21];
cx q[44], q[47];
cx q[8], q[10];
cx q[32], q[36];
cx q[18], q[22];
cx q[45], q[48];
cx q[33], q[37];
cx q[19], q[23];
cx q[34], q[38];
barrier q;

cx q[6], q[2];
cx q[30], q[20];
cx q[51], q[46];
cx q[16], q[9];
cx q[43], q[35];
cx q[7], q[3];
cx q[31], q[21];
cx q[52], q[47];
cx q[17], q[10];
cx q[44], q[36];
cx q[32], q[22];
cx q[18], q[11];
cx q[45], q[37];
cx q[33], q[23];
cx q[34], q[24];
barrier q;

cx q[6], q[12];
cx q[30], q[39];
cx q[51], q[54];
cx q[1], q[4];
cx q[16], q[25];
cx q[43], q[49];
cx q[7], q[13];
cx q[31], q[40];
cx q[17], q[26];
cx q[44], q[50];
cx q[8], q[14];
cx q[32], q[41];
cx q[18], q[27];
cx q[33], q[42];
cx q[19], q[28];
barrier q;

cx q[6], q[4];
cx q[30], q[25];
cx q[51], q[49];
cx q[16], q[13];
cx q[43], q[40];
cx q[7], q[5];
cx q[31], q[26];
cx q[52], q[50];
cx q[17], q[14];
cx q[44], q[41];
cx q[32], q[27];
cx q[18], q[15];
cx q[45], q[42];
cx q[33], q[28];
cx q[34], q[29];
barrier q;

cx q[6], q[9];
cx q[30], q[35];
cx q[51], q[53];
cx q[1], q[3];
cx q[16], q[21];
cx q[43], q[47];
cx q[7], q[10];
cx q[31], q[36];
cx q[17], q[22];
cx q[44], q[48];
cx q[8], q[11];
cx q[32], q[37];
cx q[18], q[23];
cx q[33], q[38];
cx q[19], q[24];
barrier q;

h q[6]; measure q[6] -> rec[144]; h q[6]; // decomposed MX
h q[30]; measure q[30] -> rec[145]; h q[30]; // decomposed MX
h q[51]; measure q[51] -> rec[146]; h q[51]; // decomposed MX
h q[1]; measure q[1] -> rec[147]; h q[1]; // decomposed MX
h q[16]; measure q[16] -> rec[148]; h q[16]; // decomposed MX
h q[43]; measure q[43] -> rec[149]; h q[43]; // decomposed MX
h q[7]; measure q[7] -> rec[150]; h q[7]; // decomposed MX
h q[31]; measure q[31] -> rec[151]; h q[31]; // decomposed MX
h q[52]; measure q[52] -> rec[152]; h q[52]; // decomposed MX
h q[17]; measure q[17] -> rec[153]; h q[17]; // decomposed MX
h q[44]; measure q[44] -> rec[154]; h q[44]; // decomposed MX
h q[8]; measure q[8] -> rec[155]; h q[8]; // decomposed MX
h q[32]; measure q[32] -> rec[156]; h q[32]; // decomposed MX
h q[18]; measure q[18] -> rec[157]; h q[18]; // decomposed MX
h q[45]; measure q[45] -> rec[158]; h q[45]; // decomposed MX
h q[33]; measure q[33] -> rec[159]; h q[33]; // decomposed MX
h q[19]; measure q[19] -> rec[160]; h q[19]; // decomposed MX
h q[34]; measure q[34] -> rec[161]; h q[34]; // decomposed MX
barrier q;

reset q[6];
reset q[30];
reset q[51];
reset q[1];
reset q[16];
reset q[43];
reset q[7];
reset q[31];
reset q[52];
reset q[17];
reset q[44];
reset q[8];
reset q[32];
reset q[18];
reset q[45];
reset q[33];
reset q[19];
reset q[34];
barrier q;

cx q[0], q[1];
cx q[12], q[16];
cx q[39], q[43];
cx q[4], q[7];
cx q[25], q[31];
cx q[49], q[52];
cx q[13], q[17];
cx q[40], q[44];
cx q[5], q[8];
cx q[26], q[32];
cx q[14], q[18];
cx q[41], q[45];
cx q[27], q[33];
cx q[15], q[19];
cx q[28], q[34];
barrier q;

cx q[2], q[1];
cx q[20], q[16];
cx q[46], q[43];
cx q[9], q[7];
cx q[35], q[31];
cx q[53], q[52];
cx q[21], q[17];
cx q[47], q[44];
cx q[10], q[8];
cx q[36], q[32];
cx q[22], q[18];
cx q[48], q[45];
cx q[37], q[33];
cx q[23], q[19];
cx q[38], q[34];
barrier q;

cx q[2], q[6];
cx q[20], q[30];
cx q[46], q[51];
cx q[9], q[16];
cx q[35], q[43];
cx q[3], q[7];
cx q[21], q[31];
cx q[47], q[52];
cx q[10], q[17];
cx q[36], q[44];
cx q[22], q[32];
cx q[11], q[18];
cx q[37], q[45];
cx q[23], q[33];
cx q[24], q[34];
barrier q;

cx q[12], q[6];
cx q[39], q[30];
cx q[54], q[51];
cx q[4], q[1];
cx q[25], q[16];
cx q[49], q[43];
cx q[13], q[7];
cx q[40], q[31];
cx q[26], q[17];
cx q[50], q[44];
cx q[14], q[8];
cx q[41], q[32];
cx q[27], q[18];
cx q[42], q[33];
cx q[28], q[19];
barrier q;

cx q[4], q[6];
cx q[25], q[30];
cx q[49], q[51];
cx q[13], q[16];
cx q[40], q[43];
cx q[5], q[7];
cx q[26], q[31];
cx q[50], q[52];
cx q[14], q[17];
cx q[41], q[44];
cx q[27], q[32];
cx q[15], q[18];
cx q[42], q[45];
cx q[28], q[33];
cx q[29], q[34];
barrier q;

cx q[9], q[6];
cx q[35], q[30];
cx q[53], q[51];
cx q[3], q[1];
cx q[21], q[16];
cx q[47], q[43];
cx q[10], q[7];
cx q[36], q[31];
cx q[22], q[17];
cx q[48], q[44];
cx q[11], q[8];
cx q[37], q[32];
cx q[23], q[18];
cx q[38], q[33];
cx q[24], q[19];
barrier q;

measure q[6] -> rec[162];
measure q[30] -> rec[163];
measure q[51] -> rec[164];
measure q[1] -> rec[165];
measure q[16] -> rec[166];
measure q[43] -> rec[167];
measure q[7] -> rec[168];
measure q[31] -> rec[169];
measure q[52] -> rec[170];
measure q[17] -> rec[171];
measure q[44] -> rec[172];
measure q[8] -> rec[173];
measure q[32] -> rec[174];
measure q[18] -> rec[175];
measure q[45] -> rec[176];
measure q[33] -> rec[177];
measure q[19] -> rec[178];
measure q[34] -> rec[179];
barrier q;

measure q[0] -> rec[180];
measure q[2] -> rec[181];
measure q[3] -> rec[182];
measure q[4] -> rec[183];
measure q[5] -> rec[184];
measure q[9] -> rec[185];
measure q[10] -> rec[186];
measure q[11] -> rec[187];
measure q[12] -> rec[188];
measure q[13] -> rec[189];
measure q[14] -> rec[190];
measure q[15] -> rec[191];
measure q[20] -> rec[192];
measure q[21] -> rec[193];
measure q[22] -> rec[194];
measure q[23] -> rec[195];
measure q[24] -> rec[196];
measure q[25] -> rec[197];
measure q[26] -> rec[198];
measure q[27] -> rec[199];
measure q[28] -> rec[200];
measure q[29] -> rec[201];
measure q[35] -> rec[202];
measure q[36] -> rec[203];
measure q[37] -> rec[204];
measure q[38] -> rec[205];
measure q[39] -> rec[206];
measure q[40] -> rec[207];
measure q[41] -> rec[208];
measure q[42] -> rec[209];
measure q[46] -> rec[210];
measure q[47] -> rec[211];
measure q[48] -> rec[212];
measure q[49] -> rec[213];
measure q[50] -> rec[214];
measure q[53] -> rec[215];
measure q[54] -> rec[216];

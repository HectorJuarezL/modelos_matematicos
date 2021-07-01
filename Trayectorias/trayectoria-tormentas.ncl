;;;;Evaluación de trayectorias, CIC IPN
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"


begin

  date = (/1512,1600,1612,1700,1712,1800,1812,1900/)
  ndate = dimsizes(date)
  sdate = sprinti("%4.0i",date)
  EXP = (/"EXP_I"/)                ; (/"EXP_I","EXP_II","EXP_III"/)
  nexp = dimsizes(EXP)
  a = addfile("wrfout_d03.nc","r") ;;;;Solo para cargar lat y long variables
  lat2d = a->XLAT(0,:,:)
  lon2d = a->XLONG(0,:,:)
  dimll = dimsizes(lat2d)
  nlat  = dimll(0)
  mlon  = dimll(1)
  slp = wrf_user_getvar(a,"slp",0)
  dims = dimsizes(slp)

  time = new(ndate,string)
  imin = new(ndate,integer)
  jmin = new(ndate,integer)
  smin = new(ndate,integer)
  fs = systemfunc("ls wrfout*00") ;;;; Cargar todas las salidas de un solo ensamble
  nfs= dimsizes(fs)
  if(nfs .ne. ndate) then
     print("+nfs+" .ne. "+ndate)
  end if

  do ifs=0,nfs-1
    f = addfile(fs(ifs)+".nc","r")
    time(ifs) = wrf_user_list_times(f)
    print(time(ifs))
    slp2d = wrf_user_getvar(f,"slp",0)
    slp1d     = ndtooned(slp2d) ;;;;convertir la SLP de la malla a un solo vector
    smin(ifs) = minind(slp1d)
    minij     = ind_resolve(ind(slp1d.eq.min(slp2d)),dims) ;;;; Indices de la malla en un vector i y otro j
    imin(ifs) = minij(0,0)
    jmin(ifs) = minij(0,1)
    print(time(ifs)+" : "+min(slp2d)+" ("+imin(ifs)+","+jmin(ifs)+")")
  end do




  wks=gsn_open_wks("ps","track")              
  gsn_define_colormap(wks,"BlGrYeOrReVi200")  

  res                     = True
  res@gsnDraw             = False             
  res@gsnFrame            = False             
  res@gsnMaximize         = True              

  res@tiMainString = "Huracan Bud"       

  WRF_map_c(a,res,0)                          
                                              
  plot = gsn_csm_map(wks,res)                 

  gsres                = True
  gsres@gsMarkerIndex  = 16                  
  gsres@gsMarkerSizeF = 0.005               
  cols                  = (/5,160,40, 190, 45, 128, 13)


  res_lines                      = True
  res_lines@gsLineThicknessF     = 3.           

  dot  = new(ndate,graphic)    
  line = new(ndate,graphic)    

  do i = 0,ndate-2
     res_lines@gsLineColor           = cols(0)
     xx=(/lon2d(imin(i),jmin(i)),lon2d(imin(i+1),jmin(i+1))/)
     yy=(/lat2d(imin(i),jmin(i)),lat2d(imin(i+1),jmin(i+1))/)
     line(i) = gsn_add_polyline(wks,plot,xx,yy,res_lines)
  end do

  lon1d = ndtooned(lon2d)
  lat1d = ndtooned(lat2d)

  do i = 0,ndate-1
     print("dot:"+lon1d(smin(i))+","+lat1d(smin(i)))
     gsres@gsMarkerColor  = cols(0)
     dot(i)=gsn_add_polymarker(wks,plot,lon1d(smin(i)),lat1d(smin(i)),gsres)
  end do

  txres               = True
  txres@txFontHeightF = 0.015
  txres@txFontColor   = cols(0)
  txid1 = new(ndate,graphic)

  do i = 0, ndate-1
     txres@txJust = "CenterRight"
     ix = smin(i) - 4
     print("Eye:"+ix)
     if(i.eq.1) then
        txres@txJust = "CenterLeft"
        ix = ix + 8
     end if
     txid1(i) = gsn_add_text(wks,plot,sdate(i),lon1d(ix),lat1d(ix),txres)
  end do

  txres@txJust = "CenterLeft"

  txid2 = new(nexp,graphic)
  pmid2 = new(nexp,graphic)
  do i = 0,nexp-1
     gsres@gsMarkerColor  = cols(i)
     txres@txFontColor    = cols(i)
     ii = ((/129,119,109/))  ; ilat
     jj = ((/110,110,110/))  ; jlon
     ji = ii*mlon+jj         ; col x row
     pmid2(i) = gsn_add_polymarker(wks,plot,lon1d(ji(i)),lat1d(ji(i)),gsres)
     txid2(i) = gsn_add_text(wks,plot,EXP(i),lon1d(ji(i)+5),lat1d(ji(i)),txres)
  end do

  draw(plot)
  frame(wks)
end
# coding: utf-8
import numpy as np
import scipy.signal


def get_only_recovery(rri, period=300):
    rri = np.array(rri)
    t = np.cumsum(rri) / 1000.0
    t -= t[0]
    return t[t <= period], rri[t <= period]


def calculate_hr(rri):
    return 60/(rri / 1000.0)


def fit_extreme_points(t, hr):
    ts = t[0], t[-1]
    hrs = hr[0], hr[-1]
    coef = np.polyfit(ts, hrs, 1)
    line = np.polyval(coef, ts)
    return ts, coef, line


def distance_line_point(coef, point):
    A, C = coef[0], coef[1]
    B = -1
    m, n = point[0], point[1]
    return abs(A * m + B * n + C) / np.sqrt(A ** 2 + B ** 2)


def distance_from_each_point_to_line(t, hr, coef):
    distances = []
    for i in range(len(t)):
        distances.append(distance_line_point(coef, (t[i], hr[i])))

    return np.array(distances)


def final_plot(**kwargs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(kwargs['t'], kwargs['hr'], color=[0.75, 0.75, 0.75])
    ax[0].plot(kwargs['t'], kwargs['hr_filt'], 'k')
    ax[0].plot(kwargs['ts'], kwargs['line'], 'k--')

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('HR (bpm)')
    ylim = ax[0].get_ylim()
    ax[0].plot([kwargs['hrrpt'], kwargs['hrrpt']],
               [ylim[0], ylim[1]], 'k--')
    ax[1].plot(kwargs['t'], kwargs['distances'], 'k')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Distance')
    plt.show()


def prepare_heart_rate(signal):
    t, rri = get_only_recovery(signal)
    hr = calculate_hr(rri)
    return t, hr


def filt_hr(t, hr):
    coef = np.polyfit(t, hr, 3)
    return np.polyval(coef, t)


def calculate_delta_hr(t, hr, distances):
    hrrpt_idx = distances.argmax()
    hr1 = np.median(hr[:5])
    hr2 = np.median(hr[hrrpt_idx-5:hrrpt_idx])
    hr3 = np.median(hr[hrrpt_idx:hrrpt_idx + 5])
    hr4 = np.median(hr[-5:])
    return (hr1 - hr2, hr3 - hr4)


def hrrpt(rec_hr, show=False):
    t, hr = prepare_heart_rate(rec_hr)
    hr_filt = scipy.signal.savgol_filter(hr, 21, 3)

    ts, coef, line = fit_extreme_points(t, hr_filt)
    distances = distance_from_each_point_to_line(t, hr_filt, coef)

    hrrpt = t[distances.argmax()]

    delta1, delta2 = calculate_delta_hr(t, hr_filt, distances)

    if show:
        final_plot(t=t, hr=hr, distances=distances,
                   hrrpt=hrrpt, line=line, ts=ts, hr_filt=hr_filt)

    return [hrrpt, delta1, delta2]
